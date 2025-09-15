from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import json
import functools
from datetime import datetime
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from odata_client import ODataClient, _is_guid
from log_config import setup_logging
from field_filter import filter_fields, resolve_allowed_fields
from db.database import get_session
from db.repository import InstructionRepository

from dotenv import load_dotenv

import math
import ast

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


FILTER_TOKEN_RE = re.compile(
    r"\b(eq|ne|gt|lt|ge|le|and|or|not|substringof|startswith|endswith)\b",
    re.IGNORECASE,
)


def _json_ready(x: Any) -> Any:
    """Convert dataclasses/pydantic/objects to JSON-serialisable structures."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_json_ready(i) for i in x]
    if isinstance(x, dict):
        return {k: _json_ready(v) for k, v in x.items()}
    if hasattr(x, "model_dump"):
        try:
            return _json_ready(x.model_dump())
        except Exception:
            pass
    try:
        from dataclasses import is_dataclass, asdict

        if is_dataclass(x):
            return _json_ready(asdict(x))
    except Exception:
        pass
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        try:
            return _json_ready(x.dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return _json_ready(vars(x))
        except Exception:
            pass
    return repr(x)


def _shorten(data: Any, limit: int = 500) -> str:
    try:
        txt = json.dumps(_json_ready(data), ensure_ascii=False)
    except Exception:
        txt = repr(data)
    if len(txt) > limit:
        return txt[:limit] + "..."
    return txt


def _unescape_strings(obj: Any) -> Any:
    """Recursively remove escape backslashes from string values."""
    if isinstance(obj, str):
        return obj.replace("\\'", "'").replace('\\"', '"')
    if isinstance(obj, dict):
        return {k: _unescape_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unescape_strings(v) for v in obj]
    return obj


def logged_tool(func):
    """Decorator to log tool calls and results."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(
            "MCP tool %s called with %s",
            func.__name__,
            _shorten({"args": args, "kwargs": kwargs}),
        )
        res = await func(*args, **kwargs)
        logger.info("MCP tool %s result %s", func.__name__, _shorten(res))
        return res

    return wrapper


class MCPServer:
    """Business layer on top of ODataClient with improved heuristics."""

    ENTITY_TYPE_PREFIX: Dict[str, str] = {
        # Catalogues
        "справочник": "Catalog_",
        "справочники": "Catalog_",
        "catalog": "Catalog_",
        "catalogs": "Catalog_",
        "каталог": "Catalog_",
        "каталоги": "Catalog_",
        # Documents
        "document": "Document_",
        "documents": "Document_",
        "документ": "Document_",
        "документы": "Document_",
        "журнал": "DocumentJournal_",
        "журналы": "DocumentJournal_",
        # Constants
        "constant": "Constant_",
        "constants": "Constant_",
        "константа": "Constant_",
        "константы": "Constant_",
        # Registers
        "план обмена": "ExchangePlan_",
        "планы обмена": "ExchangePlan_",
        "exchangeplan": "ExchangePlan_",
        "chart of accounts": "ChartOfAccounts_",
        "план счетов": "ChartOfAccounts_",
        "планы счетов": "ChartOfAccounts_",
        "chartofcalculationtypes": "ChartOfCalculationTypes_",
        "план видов расчета": "ChartOfCalculationTypes_",
        "планы видов расчета": "ChartOfCalculationTypes_",
        "chartofcharacteristictypes": "ChartOfCharacteristicTypes_",
        "план видов характеристик": "ChartOfCharacteristicTypes_",
        "регистр сведений": "InformationRegister_",
        "регистры сведений": "InformationRegister_",
        "informationregister": "InformationRegister_",
        "регистр накопления": "AccumulationRegister_",
        "регистры накопления": "AccumulationRegister_",
        "accumulationregister": "AccumulationRegister_",
        "регистр расчета": "CalculationRegister_",
        "регистры расчета": "CalculationRegister_",
        "calculationregister": "CalculationRegister_",
        "регистр бухгалтерии": "AccountingRegister_",
        "регистры бухгалтерии": "AccountingRegister_",
        "accountingregister": "AccountingRegister_",
        "бизнес процесс": "BusinessProcess_",
        "бизнес процессы": "BusinessProcess_",
        "businessprocess": "BusinessProcess_",
        "задача": "Task_",
        "задачи": "Task_",
        "task": "Task_",
        "tasks": "Task_",
    }

    FIELD_SYNONYMS: Dict[str, List[str]] = {
        "наименование": ["Description", "Наименование", "Name"],
        "имя": ["Description", "Name", "Наименование"],
        "описание": ["Description", "Наименование"],
        "code": ["Code", "Код"],
        "код": ["Code", "Код"],
        "артикул": ["Артикул", "SKU", "Code"],
        "инн": ["ИНН", "Inn", "INN"],
        "номер": ["Номер", "Number", "НомерДокумента", "DocumentNumber"],
        "ид": ["Ref_Key", "ID", "RefKey"],
        "guid": ["Ref_Key"],
        "гид": ["Ref_Key"],
        "количество": ["Количество", "Quantity"],
        "цена": ["Цена", "Price"],
        "сумма": ["Сумма", "Amount"],
        "стоимость": ["Сумма", "Amount", "Цена"],
        "дата": ["Дата", "Date", "ДатаДокумента"],
        "дата документа": ["Дата", "ДатаДокумента", "Date"],
        "формат": ["Формат", "Format"],
    }

    def __init__(
            self,
            base_url: str,
            username: Optional[str] = None,
            password: Optional[str] = None,
            timeout: int = 30,
            verify_ssl: bool = False,
    ) -> None:
        self.client = ODataClient(
            base_url, username=username, password=password, timeout=timeout, verify_ssl=verify_ssl
        )
        # Warm up metadata cache early to stabilise subsequent calls
        try:
            self.client.get_metadata()
            logger.info("1C OData metadata cache warmed up successfully")
        except Exception as e:
            logger.warning("Metadata warmup failed: %s", e)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: Union[Dict[str, Any], List[str], str, None]) -> Optional[str]:
        """Build $filter from dict/list/str."""
        if filters is None:
            return None
        if isinstance(filters, str):
            return filters
        if isinstance(filters, list):
            return " and ".join([f for f in filters if f])
        if isinstance(filters, dict):
            exprs: List[str] = []
            for key, value in filters.items():
                if value is None:
                    exprs.append(f"{key} eq null")
                elif isinstance(value, bool):
                    exprs.append(f"{key} eq {'true' if value else 'false'}")
                elif isinstance(value, (int, float)):
                    exprs.append(f"{key} eq {value}")
                elif isinstance(value, str):
                    if re.search(r"\s(and|or|eq|ne|gt|lt|ge|le)\s", value, re.IGNORECASE):
                        exprs.append(f"{key} {value}")
                    elif re.fullmatch(r"guid'[0-9a-fA-F-]{36}'", value):
                        exprs.append(f"{key} eq {value}")
                    elif _is_guid(value):
                        exprs.append(f"{key} eq guid'{value}'")
                    else:
                        safe = value.replace("'", "''")
                        exprs.append(f"{key} eq '{safe}'")
                else:
                    safe = str(value).replace("'", "''")
                    exprs.append(f"{key} eq '{safe}'")
            return " and ".join(exprs)
        return None

    @staticmethod
    def _result_from_response(
            response, client: ODataClient, object_name: Optional[str] = None
    ) -> Dict[str, Any]:
        data = response.values() if hasattr(response, "values") else None
        if object_name:
            data = filter_fields(object_name, data)
        return {
            "http_code": client.get_http_code(),
            "http_message": client.get_http_message(),
            "odata_error_code": client.get_error_code(),
            "odata_error_message": client.get_error_message(),
            "last_id": client.get_last_id(),
            "data": _json_ready(data),
        }

    # ------------------------------------------------------------------
    # Metadata & resolvers
    # ------------------------------------------------------------------
    def get_server_entity_sets(self) -> Dict[str, Any]:
        try:
            meta = self.client.get_metadata()
        except Exception as e:
            logger.warning("Failed to fetch metadata: %s", e)
            return {
                "http_code": self.client.get_http_code(),
                "http_message": str(e),
                "entity_sets": [],
            }
        return {
            "http_code": self.client.get_http_code(),
            "http_message": self.client.get_http_message(),
            "entity_sets": list((meta or {}).keys()),
        }

    def get_server_metadata(self) -> Dict[str, Any]:
        try:
            meta = self.client.get_metadata()
        except Exception as e:
            logger.warning("Failed to fetch metadata: %s", e)
            return {
                "http_code": self.client.get_http_code(),
                "http_message": str(e),
                "entity_sets": {},
            }
        # simplify properties to just a list of field names
        simplified: Dict[str, Dict[str, Any]] = {}
        for name, info in (meta or {}).items():
            props = info.get("properties", {})
            allowed = resolve_allowed_fields(name)
            if allowed:
                props = {k: v for k, v in props.items() if k in allowed}
            simplified[name] = {
                "entity_type": info.get("entity_type"),
                "properties": list(props.keys()),
            }

        return {
            "http_code": self.client.get_http_code(),
            "http_message": self.client.get_http_message(),
            "entity_sets": simplified,
        }

    def get_entity_schema(self, object_name: str) -> Optional[Dict[str, Any]]:
        try:
            meta = self.client.get_metadata()
        except Exception as e:
            logger.warning("Failed to fetch metadata: %s", e)
            return None
        schema = (meta or {}).get(object_name)
        if not schema:
            return None
        props = schema.get("properties", {})
        nav_props = schema.get("navigation_properties", {})
        allowed = resolve_allowed_fields(object_name)
        if allowed:
            props = {k: v for k, v in props.items() if k in allowed}
            nav_props = {k: v for k, v in nav_props.items() if k in allowed}
        schema = dict(schema)
        schema["properties"] = props
        schema["navigation_properties"] = nav_props
        return schema

    def resolve_entity_name(self, user_entity: str, user_type: Optional[str] = None) -> Optional[str]:
        if not user_entity:
            return None
        normalized = re.sub(r"\s+", "", user_entity).lower()
        prefixes: List[str] = []
        if user_type:
            pfx = self.ENTITY_TYPE_PREFIX.get(user_type.strip().lower())
            if pfx:
                prefixes = [pfx]
        if not prefixes:
            prefixes = list(set(self.ENTITY_TYPE_PREFIX.values()))
            prefixes.append("")
        try:
            meta = self.client.get_metadata()
        except Exception as e:
            logger.warning("Failed to fetch metadata: %s", e)
            return None
        entity_sets = list((meta or {}).keys())
        candidates: List[Tuple[str, str, int]] = []
        for es in entity_sets:
            for p in prefixes:
                if p and not es.startswith(p):
                    continue
                suffix = es[len(p):] if p else es
                s_norm = re.sub(r"\s+", "", suffix).lower()
                if s_norm == normalized:
                    return es
                if normalized in s_norm:
                    candidates.append((es, p, abs(len(s_norm) - len(normalized))))
        if not candidates and user_type:
            for es in entity_sets:
                suffix = es
                s_norm = re.sub(r"\s+", "", suffix).lower()
                if s_norm == normalized:
                    return es
                if normalized in s_norm:
                    candidates.append((es, "", abs(len(s_norm) - len(normalized))))
        if candidates:
            candidates.sort(key=lambda x: (1 if not x[1] else 0, x[2], x[0]))
            return candidates[0][0]
        return None

    def resolve_field_name(
            self,
            object_name: str,
            user_field: str,
            *,
            fallback: bool = True,
    ) -> Optional[str]:
        """Resolve a human readable field name to the exact OData property name.

        Parameters
        ----------
        object_name: str
            Exact name of the EntitySet.
        user_field: str
            Human readable field name provided by the user.
        fallback: bool, default True
            Whether to fall back to a default text field (``Description``/
            ``Наименование``/``Name``) when no direct match is found.  When
            ``False`` the function returns ``None`` if the field cannot be
            resolved, allowing the caller to keep the original field name and
            avoid accidental overwrites of unrelated properties.
        """
        if not user_field:
            return None
        schema = self.get_entity_schema(object_name)
        if not schema:
            return None
        props: Dict[str, Dict[str, Any]] = schema.get("properties", {})
        keys = list(props.keys())
        key_lower = user_field.strip().lower()
        if key_lower in self.FIELD_SYNONYMS:
            for cand in self.FIELD_SYNONYMS[key_lower]:
                if cand in props:
                    return cand
        for p in keys:
            if p.lower() == key_lower:
                return p
        for p in keys:
            pl = p.lower()
            if key_lower in pl or pl in key_lower:
                return p
        if fallback:
            for default_field in ["Description", "Наименование", "Name"]:
                if default_field in props:
                    return default_field
        return None

    # ------------------------------------------------------------------
    # Internal helpers for progressive search
    # ------------------------------------------------------------------
    def _resolve_reference(self, ref_spec: Any) -> Optional[str]:
        if not isinstance(ref_spec, dict):
            return None
        utype = ref_spec.get("user_type")
        uent = ref_spec.get("user_entity")
        ufilters = ref_spec.get("filters")
        top = ref_spec.get("top", 1)
        if not (utype and uent):
            return None
        found = self.search_object(utype, uent, ufilters, top=top)
        data = found.get("data")
        if isinstance(data, dict):
            return data.get("Ref_Key")
        if isinstance(data, list) and data:
            return data[0].get("Ref_Key")
        return None

    def _resolve_refs_in_payload(self, object_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        _ = self.get_entity_schema(object_name)
        out: Dict[str, Any] = {}
        for k, v in (payload or {}).items():
            fld = self.resolve_field_name(object_name, k, fallback=False) or k
            if (fld.endswith("_Key") or fld.endswith("Key")) and isinstance(v, dict):
                guid = self._resolve_reference(v)
                out[fld] = guid or v
            else:
                out[fld] = v
        return out

    def _validate_expand(self, object_name: str, expand: Optional[str]) -> Optional[str]:
        if not expand:
            return None
        schema = self.get_entity_schema(object_name) or {}
        nav_props = schema.get("navigation_properties") or {}
        nav_names = set(nav_props.keys()) if isinstance(nav_props, dict) else set(nav_props or [])
        valid: List[str] = []
        for segment in expand.split(","):
            seg = segment.strip()
            if not seg:
                continue
            root = seg.split("/", 1)[0]
            resolved = self.resolve_field_name(object_name, root, fallback=False) or root
            if resolved in nav_names:
                if resolved != root:
                    seg = seg.replace(root, resolved, 1)
                valid.append(seg)
            else:
                logger.warning(
                    "Navigation property '%s' not found for %s; skipping from expand",
                    root,
                    object_name,
                )
        return ",".join(valid) if valid else None

    def _exec_get(
            self, object_name: str, flt: Optional[str], top: Optional[int], expand: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        builder = getattr(self.client, object_name)
        expand = self._validate_expand(object_name, expand)
        if expand:
            builder = builder.expand(expand)
        if top is not None:
            builder = builder.top(int(top))
        if flt:
            builder = builder.filter(flt)
        try:
            resp = builder.get()
        except Exception as e:
            logger.warning("GET failed for %s with %s", object_name, e)
            return [], {
                "http_code": self.client.get_http_code(),
                "http_message": self.client.get_http_message() or str(e),
                "odata_error_code": self.client.get_error_code(),
                "odata_error_message": self.client.get_error_message(),
                "last_id": self.client.get_last_id(),
                "data": None,
            }
        vals = resp.values() or []
        vals = filter_fields(object_name, vals)
        return vals, self._result_from_response(resp, self.client, object_name)

    def _has_isfolder(self, object_name: str) -> bool:
        schema = self.get_entity_schema(object_name) or {}
        props = (schema.get("properties") or {})
        return "IsFolder" in props

    def _default_text_field(self, object_name: str) -> Optional[str]:
        # prefer Description/Наименование/Name or first string property
        for cand in ["Description", "Наименование", "Name"]:
            fld = self.resolve_field_name(object_name, cand)
            if fld:
                return fld
        schema = self.get_entity_schema(object_name) or {}
        props: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
        for pname, pinfo in props.items():
            if (pinfo or {}).get("type", "").endswith("String"):
                return pname
        return (props and list(props.keys())[0]) or None

    def _looks_like_filter(self, expr: str) -> bool:
        """Return True if the string resembles a ready OData filter expression."""
        if not isinstance(expr, str):
            return False
        return bool(FILTER_TOKEN_RE.search(expr))

    def _compose_expr_eq(self, field: str, value: str) -> str:
        safe = (value or "").replace("'", "''")
        return f"{field} eq '{safe}'"

    def _compose_expr_substr(self, field: str, value: Any) -> str:
        """Строит выражение substringof с константой в первом аргументе.

        OData в 1С ожидает, что первым аргументом функции будет подстрока,
        а вторым — имя поля. Именно поэтому ранее возникала ошибка
        "Неправильный тип аргумента": мы передавали аргументы в обратном
        порядке. Здесь мы формируем корректное выражение
        substringof('<подстрока>', Field).
        """
        safe = str(value or "").replace("'", "''")
        return f"substringof('{safe}', {field})"

    def _progressive_attempts_for_string(
            self, object_name: str, field: str, value: str, include_only_elements: bool
    ) -> List[str]:
        attempts = [self._compose_expr_eq(field, value)]
        schema = self.get_entity_schema(object_name) or {}
        props: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
        if props.get(field, {}).get("type") == "Edm.String":
            attempts.append(self._compose_expr_substr(field, value))
        if include_only_elements and self._has_isfolder(object_name):
            attempts = [f"{a} and IsFolder eq false" for a in attempts]
        return attempts

    def _progressive_attempts_for_dict(
            self, object_name: str, filters: Dict[str, Any], include_only_elements: bool
    ) -> List[str]:
        """
        Сначала точные eq для всех полей.
        Затем для каждого строкового поля формируем вариант с substringof,
        остальные поля остаются как eq. Собираем выражение через AND.
        """
        base_eq = self._build_filter(filters) or ""
        attempts = [base_eq] if base_eq else []

        schema = self.get_entity_schema(object_name) or {}
        props: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
        string_fields: List[Tuple[str, Any]] = []
        for k, v in filters.items():
            if props.get(k, {}).get("type") == "Edm.String":
                string_fields.append((k, v))

        for fld, val in string_fields:
            parts: List[str] = []
            for k, v in filters.items():
                if k == fld:
                    parts.append(self._compose_expr_substr(k, v))
                else:
                    parts.append(self._build_filter({k: v}) or "")
            attempts.append(" and ".join([p for p in parts if p]))

        if include_only_elements and self._has_isfolder(object_name):
            attempts = [f"{a} and IsFolder eq false" if a else "IsFolder eq false" for a in attempts]

        seen = set()
        uniq: List[str] = []
        for a in attempts:
            a = a.strip()
            if not a or a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        return uniq

    # ------------------------------------------------------------------
    # CRUD / actions
    # ------------------------------------------------------------------
    def list_objects(
            self,
            object_name: str,
            filters: Optional[Union[str, Dict[str, Any], List[str]]] = None,
            top: Optional[int] = None,
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        builder = getattr(self.client, object_name)
        expand = self._validate_expand(object_name, expand)
        if expand:
            builder = builder.expand(expand)
        if top is not None:
            builder = builder.top(int(top))
        if filters:
            flt = self._build_filter(filters)
            if flt:
                builder = builder.filter(flt)
        try:
            response = builder.get()
        except Exception as e:
            logger.warning("Failed to list objects for %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        return self._result_from_response(response, self.client, object_name)

    def find_object(
            self,
            object_name: str,
            filters: Optional[Union[str, Dict[str, Any], List[str]]] = None,
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        builder = getattr(self.client, object_name)
        expand = self._validate_expand(object_name, expand)
        if expand:
            builder = builder.expand(expand)
        if filters:
            flt = self._build_filter(filters)
            if flt:
                builder = builder.filter(flt)
        builder = builder.top(1)
        try:
            response = builder.get()
        except Exception as e:
            logger.warning("Failed to find object for %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        res = self._result_from_response(response, self.client, object_name)
        items = res.get("data") or []
        res["data"] = items[0] if items else None
        return res

    def create_object(
            self,
            object_name: str,
            data: Dict[str, Any],
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        builder = getattr(self.client, object_name)
        expand = self._validate_expand(object_name, expand)
        if expand:
            builder = builder.expand(expand)
        resolved = self._resolve_refs_in_payload(object_name, data or {})
        try:
            response = builder.create(resolved)
        except Exception as e:
            logger.warning("Failed to create object %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        return self._result_from_response(response, self.client, object_name)

    def update_object(
            self,
            object_name: str,
            object_id: Union[str, Dict[str, str]],
            data: Dict[str, Any],
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        builder = getattr(self.client, object_name).id(object_id)
        expand = self._validate_expand(object_name, expand)
        if expand:
            builder = builder.expand(expand)
        resolved = self._resolve_refs_in_payload(object_name, data or {})
        try:
            response = builder.update(data=resolved)
        except Exception as e:
            logger.warning("Failed to update object %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        return self._result_from_response(response, self.client, object_name)

    def delete_object(
            self,
            object_name: str,
            object_id: Union[str, Dict[str, str]],
            physical_delete: bool = False,
    ) -> Dict[str, Any]:
        if physical_delete:
            builder = getattr(self.client, object_name).id(object_id)
            try:
                response = builder.delete()
            except Exception as e:
                logger.warning("Failed to physically delete object %s: %s", object_name, e)
                self.client.http_code = None
                self.client.http_message = str(e)
                self.client.odata_code = None
                self.client.odata_message = None
                return {
                    "http_code": None,
                    "http_message": str(e),
                    "odata_error_code": None,
                    "odata_error_message": None,
                    "last_id": None,
                    "data": None,
                }
            return self._result_from_response(response, self.client, object_name)
        return self.update_object(object_name, object_id, {"DeletionMark": True})

    def post_document(self, object_name: str, object_id: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        builder = getattr(self.client, object_name).id(object_id)
        try:
            response = builder("Post")
        except Exception as e:
            logger.warning("Failed to post document %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        return self._result_from_response(response, self.client, object_name)

    def unpost_document(self, object_name: str, object_id: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        builder = getattr(self.client, object_name).id(object_id)
        try:
            response = builder("Unpost")
        except Exception as e:
            logger.warning("Failed to unpost document %s: %s", object_name, e)
            self.client.http_code = None
            self.client.http_message = str(e)
            self.client.odata_code = None
            self.client.odata_message = None
            return {
                "http_code": None,
                "http_message": str(e),
                "odata_error_code": None,
                "odata_error_message": None,
                "last_id": None,
                "data": None,
            }
        return self._result_from_response(response, self.client, object_name)

    def get_schema(self, object_name: str) -> Dict[str, Any]:
        meta = None
        try:
            meta = _server.client.get_metadata()
        except Exception as e:
            return {
                "http_code": _server.client.get_http_code(),
                "http_message": str(e),
                "odata_error_code": _server.client.get_error_code(),
                "odata_error_message": _server.client.get_error_message(),
                "fields": [],
            }
        schema = (meta or {}).get(object_name, {}) or {}
        props = schema.get("properties") or {}
        allowed = resolve_allowed_fields(object_name)
        if allowed:
            props = {k: v for k, v in props.items() if k in allowed}
        fields = list(props.keys())
        return {
            "http_code": _server.client.get_http_code(),
            "http_message": _server.client.get_http_message(),
            "odata_error_code": _server.client.get_error_code(),
            "odata_error_message": _server.client.get_error_message(),
            "fields": fields,
        }

    # ------------------------------------------------------------------
    # High level operations
    # ------------------------------------------------------------------
    def search_object(
            self,
            user_type: str,
            user_entity: str,
            user_filters: Optional[Union[str, Dict[str, Any], List[str]]] = None,
            top: Optional[int] = 1,
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Прогрессивный поиск.

        - filters=str:
            * если строка похожа на готовый OData `$filter`, он выполняется как есть;
            * иначе ищем по основному текстовому полю (eq → substringof), затем
              варианты с `IsFolder eq false`, если есть).
        - filters=dict: сначала точные `eq`, затем для строковых полей добавляется
          вариант с `substringof` (остальные остаются `eq`), плюс попытки с
          `IsFolder eq false`.

        Возвращает первый непустой результат (учитывая `top`).
        """
        logger.info(
            "search_object: type=%s entity=%s filters_type=%s top=%s expand=%s filters=%r",
            user_type,
            user_entity,
            type(user_filters).__name__,
            top,
            expand,
            user_filters,
        )
        object_name = self.resolve_entity_name(user_entity, user_type)
        if not object_name:
            return {
                "http_code": None,
                "http_message": None,
                "odata_error_code": None,
                "odata_error_message": f"Could not resolve entity '{user_entity}' of type '{user_type}'",
                "last_id": None,
                "data": None,
            }

        attempts: List[str] = []

        def exec_attempts(attempt_list: List[str]) -> Dict[str, Any]:
            nonlocal top, expand, object_name
            res: Dict[str, Any] = {}
            for flt in attempt_list:
                logger.debug("attempt filter: %s", flt)
                vals, res = self._exec_get(object_name, flt, top, expand)
                logger.debug(
                    "attempt result: http=%s odata=%s rows=%s",
                    res.get("http_code"),
                    res.get("odata_error_code"),
                    len(vals or []),
                )
                if vals:
                    # Если top<=1 — вернуть одиночный объект, иначе список
                    if top is not None and int(top) <= 1:
                        res["data"] = vals[0]
                    else:
                        res["data"] = vals
                    return res
                if res.get("http_code") == 400 and res.get("odata_error_code") == 21:
                    continue
            # если все пусто — вернём последний res (или “пусто”)
            if attempt_list:
                return res
            # вообще без попыток — как list_objects без фильтра
            return self.list_objects(object_name, filters=None, top=top, expand=expand)

        # Словарные фильтры: сначала нормализуем имена полей
        if isinstance(user_filters, dict):
            normalized: Dict[str, Any] = {}
            for k, v in user_filters.items():
                field = self.resolve_field_name(object_name, k) or k
                normalized[field] = v
            # 1-я волна: точные eq
            attempts = [self._build_filter(normalized) or ""]
            # 2-я волна: для каждого строкового поля — substringof
            attempts += self._progressive_attempts_for_dict(object_name, normalized, include_only_elements=False)
            # Пробуем
            res = exec_attempts([a for a in attempts if a])
            if res.get("data"):
                return res
            # Если пусто — добавим «только элементы»
            if self._has_isfolder(object_name):
                attempts_is = self._progressive_attempts_for_dict(
                    object_name, normalized, include_only_elements=True
                )
                res = exec_attempts(attempts_is)
                return res

            return res  # пусто

        # Строковый поиск или готовый фильтр
        if isinstance(user_filters, str) and user_filters:
            # Если строка похожа на готовое OData-выражение — используем её напрямую
            if self._looks_like_filter(user_filters):
                if top is not None and int(top) <= 1:
                    return self.find_object(object_name, filters=user_filters, expand=expand)
                return self.list_objects(object_name, filters=user_filters, top=top, expand=expand)

            # Иначе считаем, что это текст для поиска по основному полю
            fld = self._default_text_field(object_name)
            if not fld:
                # Как fallback — просто top(1) без фильтра
                return self.find_object(object_name, filters=None, expand=expand)
            # 1-я волна: eq → substr → substr_ci
            attempts = self._progressive_attempts_for_string(
                object_name, fld, user_filters, include_only_elements=False
            )
            res = exec_attempts(attempts)
            if res.get("data"):
                return res
            # 2-я волна: повтор с IsFolder eq false (если поле есть в схеме)
            if self._has_isfolder(object_name):
                attempts2 = self._progressive_attempts_for_string(
                    object_name, fld, user_filters, include_only_elements=True
                )
                res = exec_attempts(attempts2)
                return res
            return res

        # Иначе — как обычный вызов
        if top is not None and int(top) <= 1:
            return self.find_object(object_name, filters=user_filters, expand=expand)
        return self.list_objects(object_name, filters=user_filters, top=top, expand=expand)

    def ensure_entity(
            self,
            user_type: str,
            user_entity: str,
            data_or_filters: Union[Dict[str, Any], str],
            expand: Optional[str] = None,
    ) -> Dict[str, Any]:
        object_name = self.resolve_entity_name(user_entity, user_type)
        if not object_name:
            return {"http_code": None, "odata_error_message": f"Unknown entity {user_entity}", "data": None}
        filters = data_or_filters
        found = self.find_object(object_name, filters=filters, expand=expand)
        if found.get("data"):
            return found
        data = data_or_filters if isinstance(data_or_filters, dict) else {}
        created = self.create_object(object_name, data, expand=expand)
        return created

    def create_document_with_rows(
            self,
            object_name: str,
            header: Dict[str, Any],
            rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
            post: bool = False,
    ) -> Dict[str, Any]:
        created = self.create_object(object_name, header or {})
        if not (200 <= (created.get("http_code") or 0) < 300) or not created.get("last_id"):
            return {"step": "create_header", **created}
        doc_id = created["last_id"]
        result: Dict[str, Any] = {"header": created, "table_parts": {}}
        if rows:
            for tp_name, tp_rows in (rows or {}).items():
                builder = getattr(self.client, object_name).id(doc_id)
                tp_endpoint = getattr(builder, tp_name)
                tp_results: List[Dict[str, Any]] = []
                for row in (tp_rows or []):
                    resolved_row = self._resolve_refs_in_payload(f"{object_name}_{tp_name}", row)
                    try:
                        resp = tp_endpoint.create(resolved_row)
                    except Exception as e:
                        logger.warning("Failed to add row to %s: %s", tp_name, e)
                        tp_results.append(
                            {
                                "http_code": None,
                                "http_message": str(e),
                                "odata_error_code": None,
                                "odata_error_message": None,
                                "data": None,
                            }
                        )
                        continue
                    tp_results.append(
                        self._result_from_response(
                            resp, self.client, f"{object_name}_{tp_name}"
                        )
                    )
                result["table_parts"][tp_name] = tp_results
        if post:
            result["post"] = self.post_document(object_name, doc_id)
        return _json_ready(result)


# ------------------------------------------------------------------
# FastMCP tools
# ------------------------------------------------------------------

BASE_URL = os.getenv("MCP_1C_BASE", "")
USERNAME = os.getenv("ONEC_USERNAME")
PASSWORD = os.getenv("ONEC_PASSWORD")
VERIFY_SSL = os.getenv("ONEC_VERIFY_SSL", "false").lower() not in {"false", "0", "no"}

if "_server_instance" not in globals():
    _server_instance = MCPServer(base_url=BASE_URL,
                                 username=USERNAME,
                                 password=PASSWORD,
                                 verify_ssl=VERIFY_SSL)
_server = _server_instance
mcp = FastMCP("mcp_1c_improved", port=4200, host="0.0.0.0")


def _fetch_instruction(entity: str, action: str):
    sess_gen = get_session()
    session = next(sess_gen)
    try:
        repo = InstructionRepository(session)
        return repo.get(entity, action)
    finally:
        try:
            sess_gen.close()
        except Exception:
            try:
                session.close()
            except Exception:
                pass


@logged_tool
@mcp.tool()
async def howto(
    entity: Annotated[
        str,
        Field(
            description=(
                "Entity name or phrase (e.g. Catalog_Контрагенты or "
                "'справочник контрагенты')"
            ),
            max_length=256,
        ),
    ],
    action: Annotated[
        str,
        Field(
            description=(
                "Operation: search|get|create|update|delete|post|unpost"
            ),
            examples=["search", "create", "update", "delete", "post", "unpost"],
        ),
    ],
    user_args: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description=(
                "Optional user-provided args to consider when planning steps"
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Возвращает инструкцию из БД для указанной сущности и действия.
    Инструкция содержит описание, возможные шаги, схему аргументов и
    карту полей. Эти данные необходимо использовать для подготовки шагов todo.

    Обрабатываются случаи отсутствия инструкции и отсутствия шагов.
    """
    resolved = await asyncio.to_thread(
        _server.resolve_entity_name,
        entity,
        None,
    )

    entity_key = resolved or entity

    act = action.strip().lower()
    if act == "get":
        act = "search"

    instr = await asyncio.to_thread(_fetch_instruction, entity_key, act)

    if not instr:
        return _json_ready(
            {
                "entity_input": entity,
                "entity_resolved": resolved,
                "action": act,
                "found": False,
                "message": (
                    "Instruction not found for entity "
                    f"'{entity_key}' and action '{act}'"
                ),
            }
        )

    return _json_ready(
        {
            "entity_input": entity,
            "entity_resolved": resolved,
            "action": act,
            "found": True,
            "instruction": {
                "descr": instr.descr,
                "steps": instr.steps,
                "arg_schema": instr.arg_schema,
                "field_map": instr.field_map,
            },
            "hint": (
                "Сопоставьте user_args с field_map и вызовите todo. Если "
                "шагов нет — ответ сформируйте по descr."
            ),
        }
    )


def _ensure_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _run_todo_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single todo step against OData via MCPServer methods."""
    func = (step.get("func") or "").strip().lower()
    args = step.get("args") or {}
    if func in {"get", "search"}:
        object_name = args.get("entity_name")
        user_filters = args.get("user_filters")
        top = args.get("top")
        expand = args.get("expand")
        if top is not None:
            try:
                top = int(top)
            except Exception:
                top = None
        if top is not None and top <= 1:
            return _server.find_object(object_name, filters=user_filters, expand=expand)
        return _server.list_objects(object_name, filters=user_filters, top=top, expand=expand)

    if func == "create":
        object_name = args.get("entity_name")
        data = args.get("data") or {}
        expand = args.get("expand")
        return _server.create_object(object_name, data, expand=expand)

    if func == "update":
        object_name = args.get("entity_name")
        object_id = args.get("object_id") or args.get("id") or args.get("key")
        data = args.get("data") or {}
        expand = args.get("expand")
        return _server.update_object(object_name, object_id, data, expand=expand)

    if func == "delete":
        object_name = args.get("entity_name")
        object_id = args.get("object_id") or args.get("id") or args.get("key")
        physical_delete = _ensure_bool(args.get("physical_delete")) or False
        return _server.delete_object(object_name, object_id, physical_delete=physical_delete)

    if func == "post":
        object_name = args.get("entity_name")
        object_id = args.get("object_id") or args.get("id") or args.get("key")
        return _server.post_document(object_name, object_id)

    if func == "unpost":
        object_name = args.get("entity_name")
        object_id = args.get("object_id") or args.get("id") or args.get("key")
        return _server.unpost_document(object_name, object_id)

    return {
        "http_code": None,
        "http_message": f"Неподдерживаемая операция '{func}'",
        "odata_error_code": None,
        "odata_error_message": None,
        "data": None,
    }


@logged_tool
@mcp.tool()
async def todo(
    steps: Annotated[
        Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
        Field(
            description=(
                "Шаг или массив шагов (func + args), подготовленных по "
                "инструкции howto"
            )
        ),
    ] = None,
    instruction: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description=(
                "Опционально: объект инструкции из howto. Используется, "
                "если шаги отсутствуют — тогда будет возвращён ответ на "
                "основе описания (descr)."
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Выполняет шаги OData: search|get|create|update|delete|post|unpost.
    Возвращает статусы по шагам и сводку
    (completed|failed|no_steps).
    """
    allowed = {
        "get",
        "search",
        "create",
        "update",
        "delete",
        "post",
        "unpost",
    }

    if steps is None and instruction:
        instr = instruction.get("instruction") or instruction
        descr = (instr or {}).get("descr")
        insteps = (instr or {}).get("steps")
        if not insteps:
            return _json_ready(
                {
                    "steps": [],
                    "summary": {
                        "outcome": "no_steps",
                        "message": descr or "Нет шагов.",
                    },
                }
            )
        steps = insteps

    if isinstance(steps, dict):
        steps_list = [steps]
        step_ids = [steps.get("id") or "Step1"]
    else:
        steps_list = list(steps or [])
        step_ids = []
        if isinstance(steps, list):
            step_ids = [f"Step{i+1}" for i in range(len(steps_list))]

    results: List[Dict[str, Any]] = []
    all_ok = True
    for idx, st in enumerate(steps_list):
        sid = (
            st.get("id")
            or (
                step_ids[idx]
                if idx < len(step_ids)
                else f"Step{idx+1}"
            )
        )
        func = (st.get("func") or "").strip().lower()
        if func not in allowed:
            results.append({
                "id": sid,
                "status": "failed",
                "details": {
                    "error": f"Неподдерживаемая операция '{func}'",
                },
            })
            all_ok = False
            continue
        res = await asyncio.to_thread(_run_todo_step, st)
        http_code = res.get("http_code")
        ok = isinstance(http_code, int) and 200 <= http_code < 300
        results.append({
            "id": sid,
            "status": "completed" if ok else "failed",
            "details": _json_ready(res),
        })
        all_ok = all_ok and ok

    outcome = "completed" if all_ok else "failed"
    return _json_ready(
        {"steps": results, "summary": {"outcome": outcome}}
    )


@logged_tool
@mcp.tool()
async def solve(
        expression: Annotated[str,
        Field(description="Математическое выражение в виде строки",
        examples=["2+3", "4.2*5", "147/3", "56+68+72.3-10", "(19+134)**2", "log(10)", "sqrt(4)"],
        max_length=256)]
) -> Dict[str, Any]:
    """
    Возвращает результат вычисления арифметического выражения.
    Поддерживаются операции: +, -, *, /, //, **, sqrt, log, log10, exp
    Поддерживается указание порядка операций с помощью скобок

    Args:
        expression: математическое выражение в виде строки

    Returns:
        Dict с следующими полями:
        - success (bool): Флаг успешного выполнения операции
        - result (str|null): Результат вычисления арифметического выражения в виде строки
        - error (str|null): Сообщение об ошибке если операция не удалась

    Примеры использования:
       - Умножить 147.5 на 5:
        solve("147.5*5")
       - Возвести число 1440 в квадрат:
        solve("1440**2")
       - Посчитать сумму 1245, 1030, 5044, 986:
        solve("1245+1030+5044+986")
       - Посчитать десятичный логарифм от 25:
        solve("log10(25)")
    """

    def _sync() -> Dict[str, Any]:
        try:
            tree = ast.parse(expression, mode='eval')

            allowed_nodes = (ast.Expression, ast.UnaryOp, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                             ast.Pow, ast.Constant, ast.Name, ast.Call, ast.Load)
            allowed_operations = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)
            allowed_names = {
                'pi': math.pi,
                'e': math.e,
                'sqrt': math.sqrt,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
            }

            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError(f"Недопустимый узел: {type(node).__name__}")
                if isinstance(node, (ast.BinOp)):
                    if not isinstance(node.op, allowed_operations):
                        raise ValueError(f"Недопустимый бинарный оператор: {type(node.op).__name__}")
                if isinstance(node, (ast.UnaryOp)):
                    if not isinstance(node.op, allowed_operations):
                        raise ValueError(f"Недопустимый унарный оператор: {type(node.op).__name__}")
                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        raise ValueError(f"Недопустимое имя: {node.id}")
                if isinstance(node, ast.Call):
                    if node.func.id not in allowed_names:
                        raise ValueError(f"Недопустимая функция: {node.func.id}")

            code = compile(tree, '<string>', 'eval')
            res = eval(code, {"__builtins__": {}}, allowed_names)

            return {
                "success": True,
                "result": str(res),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Ошибка решения выражения: {str(e)}"
            }

    result = await asyncio.to_thread(_sync)
    return _json_ready(result)


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import sys

    mcp.run(transport="sse")
