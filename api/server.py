"""REST API over the decisioning path.

Implements the subset of IPSRS FR-API-01..03 that the prototype covers, built
on the standard library so the service starts with no installation:

  POST /v1/prequalification            indicative eligibility, no bureau pull
  POST /v1/applications/decision       full decisioning (the core endpoint)
  GET  /v1/decisions/{decision_id}     retrieve a stored decision
  GET  /v1/applications/{id}/decisions decision history for an application
  GET  /v1/audit/{correlation_id}      audit trail reconstruction
  GET  /v1/audit/verify                tamper-evidence check on the chain
  GET  /v1/usage                       metered usage and reconciliation
  GET  /v1/partners/health             partner availability and latency
  GET  /openapi.json                   machine-readable specification
  GET  /healthz                        liveness

Cross-cutting behaviour, applied in one place rather than per endpoint:

* **Authentication** by API key, resolved to exactly one tenant. Every response
  is scoped to that tenant; there is no route by which one tenant can name
  another's identifiers (IPSRS FR-ADM-06).
* **Idempotency**: a repeated ``Idempotency-Key`` returns the original decision
  rather than creating a second one - and is not billed twice.
* **Correlation**: every request carries or is assigned a correlation id, which
  is echoed in the response header and stamped on every audit and metering row.
* **Standardised errors**: a machine-readable ``code``, a human ``message`` and
  the correlation id, and never a stack trace or an internal path.

Production would put this behind an API gateway with OAuth 2.0/OIDC, mutual TLS
for institutional partners and signed webhooks (IPSRS FR-API-02). The prototype
demonstrates the contract, not the edge.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from core import batch as batch_module
from core import config, monitoring
from core.auth import AuthService, Principal
from core.ledger import Ledger
from core.pipeline import Platform

UI_PATH = pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html"

API_VERSION = "1.0.0"
MAX_BODY_BYTES = 256 * 1024

# Simple fixed-window rate limit per tenant (IPSRS FR-API-02).
RATE_LIMIT_PER_MINUTE = 600
_rate_state: dict[str, tuple[int, int]] = {}
_rate_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str,
                 details: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def _rate_limit(tenant_code: str) -> None:
    window = int(time.time() // 60)
    with _rate_lock:
        current_window, count = _rate_state.get(tenant_code, (window, 0))
        if current_window != window:
            current_window, count = window, 0
        count += 1
        _rate_state[tenant_code] = (current_window, count)
    if count > RATE_LIMIT_PER_MINUTE:
        raise ApiError(429, "RATE_LIMIT_EXCEEDED",
                       "Too many requests; retry after the current minute.")


def _require(payload: dict, path: str) -> Any:
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node or node[part] is None:
            raise ApiError(400, "VALIDATION_FAILED",
                           f"Missing required field: {path}",
                           {"field": path})
        node = node[part]
    return node


def _validate_application(application: dict) -> None:
    """Intake validation (IPSRS FR-INT-02, VAL-*): reject before any external call."""
    errors: list[dict] = []

    national_id = application.get("national_id")
    if not national_id:
        errors.append({"field": "application.national_id",
                       "error": "required"})
    elif not re.fullmatch(r"\d{6}/\d{2}/\d", str(national_id)):
        errors.append({"field": "application.national_id",
                       "error": "expected Zambian NRC format ######/##/#"})

    for field in ("date_of_birth", "application_date"):
        value = application.get(field)
        if not value:
            errors.append({"field": f"application.{field}", "error": "required"})
        elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
            errors.append({"field": f"application.{field}",
                           "error": "expected ISO 8601 date (YYYY-MM-DD)"})

    amount = application.get("requested_amount")
    if amount is None:
        errors.append({"field": "application.requested_amount",
                       "error": "required"})
    elif not isinstance(amount, (int, float)) or amount <= 0:
        errors.append({"field": "application.requested_amount",
                       "error": "must be a positive number"})

    tenor = application.get("tenor_months")
    if tenor is None:
        errors.append({"field": "application.tenor_months", "error": "required"})
    elif not isinstance(tenor, int) or tenor <= 0:
        errors.append({"field": "application.tenor_months",
                       "error": "must be a positive integer"})

    income = application.get("declared_monthly_income")
    if income is None:
        errors.append({"field": "application.declared_monthly_income",
                       "error": "required"})
    elif not isinstance(income, (int, float)) or income <= 0:
        errors.append({"field": "application.declared_monthly_income",
                       "error": "must be a positive number"})

    consent = application.get("consent") or {}
    for purpose in ("credit_bureau_enquiry", "automated_decisioning"):
        if not consent.get(purpose):
            errors.append({"field": f"application.consent.{purpose}",
                           "error": "consent required before processing"})

    if errors:
        raise ApiError(400, "VALIDATION_FAILED",
                       "The application failed intake validation.",
                       {"errors": errors})


def _reference_scores(platform: Platform, tenant: config.Tenant) -> list[float]:
    """The development-sample PD distribution, expressed on this tenant's scale."""
    from core.scorecard import pd_to_score
    reference = platform.card._a.get("reference_pd_distribution") or []
    return [float(pd_to_score(pd, tenant.score_scale)) for pd in reference]


_default_auth: Optional[AuthService] = None
_default_auth_lock = threading.Lock()


def default_auth_service() -> AuthService:
    """One shared AuthService for callers that do not supply their own.

    Password hashing is deliberately slow (PBKDF2, 240k iterations). Building a
    fresh service per API instance would pay that cost repeatedly for no
    benefit, so the default is created once.
    """
    global _default_auth
    with _default_auth_lock:
        if _default_auth is None:
            _default_auth = AuthService()
        return _default_auth


class Api:
    """Routing and handlers, independent of the HTTP server plumbing."""

    def __init__(self, platform: Optional[Platform] = None,
                 auth: Optional[AuthService] = None):
        self.platform = platform or Platform()
        self.auth = auth or default_auth_service()
        # Route -> permission required. A caller without it never reaches the
        # handler, so authorisation is not left to each handler to remember.
        self.permissions: dict[Callable, str] = {}
        self.routes: list[tuple[str, re.Pattern, Callable]] = [
            ("POST", re.compile(r"^/v1/prequalification$"), self.prequalify),
            ("POST", re.compile(r"^/v1/applications/decision$"), self.decide),
            ("GET", re.compile(r"^/v1/decisions/(?P<decision_id>[A-Za-z0-9\-]+)$"),
             self.get_decision),
            ("GET", re.compile(r"^/v1/applications/(?P<application_id>[A-Za-z0-9\-]+)/decisions$"),
             self.application_decisions),
            ("GET", re.compile(r"^/v1/audit/verify$"), self.verify_audit),
            ("GET", re.compile(r"^/v1/audit/(?P<correlation_id>[A-Za-z0-9\-]+)$"),
             self.audit_trail),
            ("GET", re.compile(r"^/v1/usage$"), self.usage),
            ("GET", re.compile(r"^/v1/partners/health$"), self.partner_health),
            ("GET", re.compile(r"^/v1/decisions$"), self.recent_decisions),
            ("GET", re.compile(r"^/v1/monitoring/summary$"), self.monitoring),
            ("POST", re.compile(r"^/v1/batches$"), self.submit_batch),
            ("POST", re.compile(r"^/v1/auth/login$"), self.login),
            ("POST", re.compile(r"^/v1/auth/logout$"), self.logout),
            ("GET", re.compile(r"^/v1/auth/session$"), self.session_info),
        ]
        self.permissions = {
            self.prequalify: "decide",
            self.decide: "decide",
            self.get_decision: "read_decisions",
            self.application_decisions: "read_decisions",
            self.recent_decisions: "read_decisions",
            self.audit_trail: "read_audit",
            self.verify_audit: "read_audit",
            self.usage: "read_usage",
            self.monitoring: "read_monitoring",
            self.partner_health: "read_partners",
            self.submit_batch: "run_batch",
        }
        #: Endpoints reachable without authentication.
        self.public = {self.login}

    # -- helpers ----------------------------------------------------------- #
    def _tenant_product(self, tenant: config.Tenant, body: dict) -> config.Product:
        code = body.get("product_code") or next(iter(tenant.products))
        if code not in tenant.products:
            raise ApiError(404, "PRODUCT_NOT_FOUND",
                           f"Product {code!r} is not configured for this tenant.",
                           {"available": sorted(tenant.products)})
        return tenant.products[code]

    # -- handlers ---------------------------------------------------------- #
    def prequalify(self, *, tenant: config.Tenant, body: dict,
                   correlation_id: str, **_: Any) -> tuple[int, dict]:
        product = self._tenant_product(tenant, body)
        application = dict(_require(body, "application"))
        income = application.get("declared_monthly_income")
        if not income:
            raise ApiError(400, "VALIDATION_FAILED",
                           "declared_monthly_income is required",
                           {"field": "application.declared_monthly_income"})

        from core import affordability as afford
        assessment = afford.assess({"application": application}, product=product)
        eligible = (assessment.max_affordable_amount >= product.min_amount)

        self.platform.ledger.meter(tenant_code=tenant.code,
                                   event_type="PREQUALIFICATION",
                                   correlation_id=correlation_id)
        self.platform.ledger.record(
            tenant_code=tenant.code, event_type="PREQUALIFICATION",
            correlation_id=correlation_id,
            payload={"eligible": eligible,
                     "indicative_maximum": str(assessment.max_affordable_amount)})
        return 200, {
            "eligible": eligible,
            "product_code": product.code,
            "currency": product.currency,
            "indicative_minimum": str(product.min_amount),
            "indicative_maximum": str(min(assessment.max_affordable_amount,
                                          product.max_amount)),
            "basis": ("affordability only; no credit bureau enquiry was "
                      "performed and no credit decision has been made"),
            "correlation_id": correlation_id,
        }

    def decide(self, *, tenant: config.Tenant, body: dict, correlation_id: str,
               idempotency_key: Optional[str], **_: Any) -> tuple[int, dict]:
        product = self._tenant_product(tenant, body)
        application = dict(_require(body, "application"))
        _validate_application(application)

        if idempotency_key:
            existing = self.platform.ledger.replay(tenant_code=tenant.code,
                                                   key=idempotency_key)
            if existing:
                stored = self.platform.ledger.get_decision(
                    tenant_code=tenant.code, decision_id=existing)
                if stored:
                    stored["replayed"] = True
                    return 200, stored

        result, telemetry = self.platform.decide(
            {"application": application,
             "internal": body.get("internal") or {},
             "partner_overrides": body.get("partner_overrides") or {}},
            tenant=tenant, product=product,
            application_id=body.get("application_id"),
            correlation_id=correlation_id)

        if idempotency_key:
            self.platform.ledger.remember_idempotency(
                tenant_code=tenant.code, key=idempotency_key,
                decision_id=result.decision_id)

        payload = result.as_dict(audience=body.get("audience", "customer"))
        payload["telemetry"] = telemetry
        return 200, payload

    def get_decision(self, *, tenant: config.Tenant, decision_id: str,
                     **_: Any) -> tuple[int, dict]:
        stored = self.platform.ledger.get_decision(tenant_code=tenant.code,
                                                   decision_id=decision_id)
        if stored is None:
            # Deliberately identical whether the decision does not exist or
            # belongs to another tenant: existence is itself information.
            raise ApiError(404, "DECISION_NOT_FOUND",
                           "No decision with that identifier for this tenant.")
        return 200, stored

    def application_decisions(self, *, tenant: config.Tenant,
                              application_id: str, **_: Any) -> tuple[int, dict]:
        decisions = self.platform.ledger.decisions_for_application(
            tenant_code=tenant.code, application_id=application_id)
        return 200, {"application_id": application_id,
                     "count": len(decisions), "decisions": decisions}

    def audit_trail(self, *, tenant: config.Tenant, correlation_id: str,
                    **_: Any) -> tuple[int, dict]:
        events = self.platform.ledger.audit_trail(
            tenant_code=tenant.code, correlation_id=correlation_id)
        return 200, {"correlation_id": correlation_id,
                     "events": events, "count": len(events)}

    def verify_audit(self, *, tenant: config.Tenant, **_: Any) -> tuple[int, dict]:
        return 200, self.platform.ledger.verify_chain(tenant_code=tenant.code)

    def usage(self, *, tenant: config.Tenant, **_: Any) -> tuple[int, dict]:
        usage = self.platform.ledger.usage(tenant_code=tenant.code)
        usage["reconciliation"] = self.platform.ledger.reconcile(
            tenant_code=tenant.code)
        return 200, usage

    def login(self, *, body: dict, correlation_id: str, **_: Any
              ) -> tuple[int, dict]:
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        user = self.auth.authenticate(username, password)
        if user is None:
            # One message for every failure mode: wrong password, unknown user
            # and locked account are indistinguishable to a caller.
            raise ApiError(401, "INVALID_CREDENTIALS",
                           "Username or password is incorrect, or the account "
                           "is temporarily locked.")
        session = self.auth.issue_session(user)
        self.platform.ledger.record(
            tenant_code=user.tenant, event_type="USER_SIGN_IN",
            actor=user.username, correlation_id=correlation_id,
            payload={"role": user.role, "method": "password"})
        return 200, session

    def logout(self, *, principal: Optional[Principal] = None,
               token: Optional[str] = None, **_: Any) -> tuple[int, dict]:
        if token:
            self.auth.revoke(token)
        if principal:
            self.platform.ledger.record(
                tenant_code=principal.tenant, event_type="USER_SIGN_OUT",
                actor=principal.username, payload={})
        return 200, {"signed_out": True}

    def session_info(self, *, principal: Principal, **_: Any
                     ) -> tuple[int, dict]:
        return 200, {
            "username": principal.username, "name": principal.display_name,
            "role": principal.role, "tenant": principal.tenant,
            "permissions": sorted(principal.permissions),
            "method": principal.method,
        }

    def partner_health(self, **_: Any) -> tuple[int, dict]:
        return 200, {"partners": self.platform.connectors.health()}

    def recent_decisions(self, *, tenant: config.Tenant, **_: Any
                         ) -> tuple[int, dict]:
        decisions = self.platform.ledger.recent_decisions(
            tenant_code=tenant.code, limit=50)
        return 200, {"count": len(decisions), "decisions": decisions}

    def monitoring(self, *, tenant: config.Tenant, **_: Any) -> tuple[int, dict]:
        """Monitoring pack computed from this tenant's stored decisions.

        The reference distribution is the scorecard's own development sample,
        so PSI answers "does live business look like what the model was built
        on?" - the question that matters, rather than a comparison against an
        arbitrary earlier week.
        """
        decisions = self.platform.ledger.recent_decisions(
            tenant_code=tenant.code, limit=1000)
        reference = _reference_scores(self.platform, tenant)
        report = monitoring.report(
            decisions=decisions, reference_scores=reference,
            model_version=self.platform.card.model_version)
        report["tenant"] = tenant.code
        report["reference"] = {
            "source": "scorecard development sample (synthetic)",
            "n": len(reference),
        }
        return 200, report

    def submit_batch(self, *, tenant: config.Tenant, body: dict,
                     correlation_id: str, **_: Any) -> tuple[int, dict]:
        product = self._tenant_product(tenant, body)
        content = body.get("csv")
        if content is None and body.get("use_sample"):
            content = batch_module.sample_file(int(body.get("rows", 40)))
        if not content:
            raise ApiError(400, "VALIDATION_FAILED",
                           "Provide a 'csv' payload or set use_sample=true.",
                           {"field": "csv"})

        result = batch_module.run(content, tenant=tenant, product=product,
                                  platform=self.platform,
                                  limit=body.get("limit"))
        self.platform.ledger.meter(
            tenant_code=tenant.code, event_type="BATCH_SCORE",
            quantity=result.processed, correlation_id=result.batch_id,
            reference=result.batch_id)
        payload = result.summary()
        payload["decisions"] = result.decisions
        payload["rejects"] = result.rejects
        return 200, payload

    # -- dispatch ---------------------------------------------------------- #
    def handle(self, method: str, path: str, headers: dict,
               body_bytes: bytes) -> tuple[int, dict, dict]:
        correlation_id = (headers.get("x-correlation-id")
                          or f"COR-{uuid.uuid4().hex[:12].upper()}")
        response_headers = {"X-Correlation-Id": correlation_id,
                            "X-Api-Version": API_VERSION}
        try:
            if path == "/healthz":
                return 200, {"status": "ok", "api_version": API_VERSION}, response_headers
            if path == "/openapi.json":
                return 200, openapi_document(), response_headers
            if path in ("/", "/ui", "/ui/"):
                return 200, {"_html": UI_PATH.read_text()}, response_headers
            if path == "/v1/demo/info":
                return 200, {
                    "environment": os.environ.get("DEMO_ENVIRONMENT",
                                                  "demonstration"),
                    "banner": os.environ.get(
                        "DEMO_BANNER",
                        "DEMONSTRATION - synthetic data, prototype model, "
                        "not approved for production use"),
                    "accounts": (self.auth.demo_accounts()
                                 if self.auth.using_default_accounts() else []),
                    "default_accounts_in_use": self.auth.using_default_accounts(),
                    "api_version": API_VERSION,
                }, response_headers

            # Two authentication routes: a session token for the console and
            # an API key for machine integration. Both resolve to a principal
            # bound to exactly one tenant.
            bearer = headers.get("authorization", "")
            token = (bearer[7:].strip() if bearer.lower().startswith("bearer ")
                     else headers.get("x-session-token"))
            principal = self.auth.validate_session(token) if token else None

            tenant = None
            if principal is not None:
                tenant = config.get_tenant(principal.tenant)
            else:
                api_key = headers.get("x-api-key")
                if api_key:
                    tenant = config.tenant_for_api_key(api_key)
                    if tenant is None:
                        raise ApiError(401, "UNAUTHENTICATED",
                                       "Unrecognised API key.")
                    # A machine integration carries the full machine scope.
                    principal = Principal(
                        username=f"api:{tenant.code}", display_name="API client",
                        role="API_CLIENT", tenant=tenant.code, method="api_key",
                        permissions={"decide", "read_decisions", "read_audit",
                                     "read_usage", "read_monitoring",
                                     "read_partners", "run_batch"})

            if tenant is not None:
                _rate_limit(tenant.code)

            if len(body_bytes) > MAX_BODY_BYTES:
                raise ApiError(413, "PAYLOAD_TOO_LARGE",
                               "Request body exceeds the permitted size.")
            body: dict = {}
            if body_bytes:
                try:
                    body = json.loads(body_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ApiError(400, "MALFORMED_JSON",
                                   "Request body is not valid JSON.")
                if not isinstance(body, dict):
                    raise ApiError(400, "MALFORMED_JSON",
                                   "Request body must be a JSON object.")

            for route_method, pattern, handler in self.routes:
                match = pattern.match(path)
                if not match:
                    continue
                if route_method != method:
                    raise ApiError(405, "METHOD_NOT_ALLOWED",
                                   f"{method} is not supported on this path.")
                # Path parameters take precedence over request-level values of
                # the same name: /v1/audit/{correlation_id} addresses a past
                # correlation id, not this request's.
                if handler not in self.public:
                    if principal is None or tenant is None:
                        raise ApiError(401, "UNAUTHENTICATED",
                                       "Sign in, or provide an API key in the "
                                       "X-API-Key header.")
                    required = self.permissions.get(handler)
                    if required and not principal.may(required):
                        raise ApiError(403, "FORBIDDEN",
                                       f"Your role ({principal.role}) does not "
                                       f"permit this action.")
                arguments: dict[str, Any] = {
                    "tenant": tenant, "body": body,
                    "correlation_id": correlation_id,
                    "idempotency_key": headers.get("idempotency-key"),
                    "principal": principal, "token": token,
                }
                arguments.update(match.groupdict())
                status, payload = handler(**arguments)
                return status, payload, response_headers

            raise ApiError(404, "NOT_FOUND", "Unknown endpoint.")

        except ApiError as error:
            return error.status, {
                "error": {"code": error.code, "message": error.message,
                          **({"details": error.details} if error.details else {})},
                "correlation_id": correlation_id,
            }, response_headers
        except Exception:
            # Never leak internals to a caller; the correlation id is the
            # thread an operator follows into the logs.
            return 500, {
                "error": {"code": "INTERNAL_ERROR",
                          "message": "The request could not be completed."},
                "correlation_id": correlation_id,
            }, response_headers


# --------------------------------------------------------------------------- #
# OpenAPI document
# --------------------------------------------------------------------------- #
def openapi_document() -> dict:
    error_schema = {
        "type": "object",
        "properties": {
            "error": {"type": "object", "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"}}},
            "correlation_id": {"type": "string"}},
    }
    security = [{"ApiKeyAuth": []}, {"SessionToken": []}]

    def response(description: str, schema: Optional[dict] = None) -> dict:
        return {"description": description,
                "content": {"application/json": {
                    "schema": schema or {"type": "object"}}}}

    standard_errors = {
        "400": response("Validation failed", error_schema),
        "401": response("Unauthenticated", error_schema),
        "404": response("Not found", error_schema),
        "429": response("Rate limit exceeded", error_schema),
    }

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Retail Lending Scoring & Decision Platform API",
            "version": API_VERSION,
            "description": (
                "Prototype API covering the core decisioning path. Decisions "
                "are produced by a scorecard trained on synthetic data and are "
                "not approved for production use."),
        },
        "servers": [{"url": "http://localhost:8080"}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header",
                               "name": "X-API-Key"},
                "SessionToken": {"type": "http", "scheme": "bearer",
                                 "description": "Session token from "
                                                "POST /v1/auth/login"}},
            "parameters": {
                "CorrelationId": {
                    "name": "X-Correlation-Id", "in": "header",
                    "required": False, "schema": {"type": "string"},
                    "description": "Echoed in the response and stamped on "
                                   "every audit and metering row."},
                "IdempotencyKey": {
                    "name": "Idempotency-Key", "in": "header",
                    "required": False, "schema": {"type": "string"},
                    "description": "Replaying a key returns the original "
                                   "decision and is not billed again."},
            },
            "schemas": {
                "Error": error_schema,
                "Consent": {
                    "type": "object",
                    "required": ["credit_bureau_enquiry", "automated_decisioning"],
                    "properties": {
                        "credit_bureau_enquiry": {"type": "boolean"},
                        "automated_decisioning": {"type": "boolean"},
                        "payroll_verification": {"type": "boolean"}},
                },
                "Application": {
                    "type": "object",
                    "required": ["national_id", "date_of_birth",
                                 "application_date", "requested_amount",
                                 "tenor_months", "declared_monthly_income",
                                 "consent"],
                    "properties": {
                        "national_id": {"type": "string",
                                        "pattern": r"^\d{6}/\d{2}/\d$",
                                        "example": "123456/78/1"},
                        "full_name": {"type": "string"},
                        "date_of_birth": {"type": "string", "format": "date"},
                        "application_date": {"type": "string", "format": "date"},
                        "employer_code": {"type": "string"},
                        "device_id": {"type": "string"},
                        "requested_amount": {"type": "number", "minimum": 0},
                        "tenor_months": {"type": "integer", "minimum": 1},
                        "declared_monthly_income": {"type": "number", "minimum": 0},
                        "declared_monthly_expenses": {"type": "number"},
                        "existing_monthly_debt_service": {"type": "number"},
                        "dependants": {"type": "integer", "minimum": 0},
                        "consent": {"$ref": "#/components/schemas/Consent"}},
                },
                "DecisionRequest": {
                    "type": "object",
                    "required": ["application"],
                    "properties": {
                        "product_code": {"type": "string"},
                        "application_id": {"type": "string"},
                        "audience": {"type": "string",
                                     "enum": ["customer", "internal"]},
                        "application": {"$ref": "#/components/schemas/Application"}},
                },
                "Decision": {
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string",
                                    "enum": ["APPROVE", "DECLINE", "REFER",
                                             "INSUFFICIENT_INFORMATION"]},
                        "decline_type": {"type": "string", "nullable": True},
                        "reason_codes": {"type": "array", "items": {
                            "type": "object", "properties": {
                                "code": {"type": "string"},
                                "text": {"type": "string"},
                                "category": {"type": "string"}}}},
                        "identifiers": {"type": "object"},
                        "assessment": {"type": "object"},
                        "versions": {"type": "object"},
                        "offer": {"type": "object"},
                        "affordability": {"type": "object", "nullable": True},
                        "trace": {"type": "object"},
                        "decided_at": {"type": "string", "format": "date-time"},
                        "expires_at": {"type": "string", "format": "date-time",
                                       "nullable": True}},
                },
            },
        },
        "security": security,
        "paths": {
            "/v1/prequalification": {"post": {
                "summary": "Indicative eligibility without a bureau enquiry",
                "parameters": [{"$ref": "#/components/parameters/CorrelationId"}],
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "$ref": "#/components/schemas/DecisionRequest"}}}},
                "responses": {"200": response("Indicative eligibility"),
                              **standard_errors}}},
            "/v1/applications/decision": {"post": {
                "summary": "Submit an application and receive a decision",
                "parameters": [
                    {"$ref": "#/components/parameters/CorrelationId"},
                    {"$ref": "#/components/parameters/IdempotencyKey"}],
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "$ref": "#/components/schemas/DecisionRequest"}}}},
                "responses": {
                    "200": response("Decision contract", {
                        "$ref": "#/components/schemas/Decision"}),
                    **standard_errors}}},
            "/v1/decisions/{decision_id}": {"get": {
                "summary": "Retrieve a stored decision",
                "parameters": [{"name": "decision_id", "in": "path",
                                "required": True, "schema": {"type": "string"}}],
                "responses": {"200": response("Decision contract", {
                    "$ref": "#/components/schemas/Decision"}),
                    **standard_errors}}},
            "/v1/applications/{application_id}/decisions": {"get": {
                "summary": "Decision history for an application",
                "parameters": [{"name": "application_id", "in": "path",
                                "required": True, "schema": {"type": "string"}}],
                "responses": {"200": response("Decision history"),
                              **standard_errors}}},
            "/v1/audit/{correlation_id}": {"get": {
                "summary": "Reconstruct the audit trail for one correlation id",
                "parameters": [{"name": "correlation_id", "in": "path",
                                "required": True, "schema": {"type": "string"}}],
                "responses": {"200": response("Audit events"),
                              **standard_errors}}},
            "/v1/audit/verify": {"get": {
                "summary": "Verify the tamper-evident audit chain",
                "responses": {"200": response("Chain verification result"),
                              **standard_errors}}},
            "/v1/usage": {"get": {
                "summary": "Metered usage and decision reconciliation",
                "responses": {"200": response("Usage statement"),
                              **standard_errors}}},
            "/v1/partners/health": {"get": {
                "summary": "Partner availability, latency and circuit state",
                "responses": {"200": response("Partner health"),
                              **standard_errors}}},
            "/v1/auth/login": {"post": {
                "summary": "Sign in and receive a session token",
                "security": [],
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "required": ["username", "password"],
                        "properties": {"username": {"type": "string"},
                                       "password": {"type": "string",
                                                    "format": "password"}}}}}},
                "responses": {
                    "200": response("Session token and user profile", {
                        "type": "object",
                        "properties": {
                            "token": {"type": "string"},
                            "expires_at": {"type": "integer"},
                            "user": {"type": "object"}}}),
                    "401": response("Invalid credentials, or the account is "
                                    "temporarily locked", error_schema)}}},
            "/v1/auth/logout": {"post": {
                "summary": "Revoke the current session token",
                "responses": {"200": response("Signed out"),
                              **standard_errors}}},
            "/v1/auth/session": {"get": {
                "summary": "Describe the authenticated caller and their permissions",
                "responses": {"200": response("Session description"),
                              **standard_errors}}},
            "/v1/demo/info": {"get": {
                "summary": "Demonstration banner and account list",
                "security": [],
                "responses": {"200": response("Demonstration metadata")}}},
            "/v1/decisions": {"get": {
                "summary": "Recent decisions for the calling tenant",
                "responses": {"200": response("Recent decisions"),
                              **standard_errors}}},
            "/v1/monitoring/summary": {"get": {
                "summary": "Model and portfolio monitoring pack (PSI, "
                           "calibration, approval rates)",
                "responses": {"200": response("Monitoring report"),
                              **standard_errors}}},
            "/v1/batches": {"post": {
                "summary": "Submit a batch file for scoring",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "csv": {"type": "string",
                                    "description": "CSV payload with a header row"},
                            "use_sample": {"type": "boolean"},
                            "rows": {"type": "integer"},
                            "product_code": {"type": "string"}}}}}},
                "responses": {"200": response("Batch summary, decisions and rejects"),
                              **standard_errors}}},
            "/healthz": {"get": {"summary": "Liveness", "security": [],
                                 "responses": {"200": response("Alive")}}},
            "/openapi.json": {"get": {"summary": "This document", "security": [],
                                      "responses": {"200": response("Spec")}}},
        },
    }


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ScoringPlatformPrototype/1.0"

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {key.lower(): value for key, value in self.headers.items()}
            path = self.path.split("?", 1)[0]
            status, payload, response_headers = api.handle(method, path,
                                                           headers, body)
            html = isinstance(payload, dict) and "_html" in payload
            encoded = (payload["_html"].encode() if html
                       else json.dumps(payload, indent=2, default=str).encode())
            self.send_response(status)
            self.send_header("Content-Type",
                             "text/html; charset=utf-8" if html
                             else "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):     # noqa: N802
            self._dispatch("GET")

        def do_POST(self):    # noqa: N802
            self._dispatch("POST")

        def log_message(self, *args):     # keep demo output readable
            return

    return Handler


def serve(port: int = 8080, *, platform: Optional[Platform] = None,
          auth: Optional[AuthService] = None,
          host: Optional[str] = None) -> ThreadingHTTPServer:
    api = Api(platform, auth)
    bind = host or os.environ.get("DEMO_BIND", "127.0.0.1")
    server = ThreadingHTTPServer((bind, port), make_handler(api))
    server.api = api
    return server


def main() -> None:
    port = int(os.environ.get("PORT", os.environ.get("DEMO_PORT", 8080)))
    server = serve(port)
    host, bound = server.server_address
    for warning in server.api.auth.warnings:
        print(f"WARNING: {warning}")
    print(f"Scoring platform listening on http://{host}:{bound}")
    print("  console      /            (sign in with a demonstration account)")
    print("  api          /v1/applications/decision")
    print("  spec         /openapi.json")
    print("  health       /healthz")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
