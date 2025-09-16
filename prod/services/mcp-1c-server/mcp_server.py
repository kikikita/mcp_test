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
from google.instructions_sync import sync_instructions_from_google

from dotenv import load_dotenv

import math
import ast

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

try:
    synced_instructions = sync_instructions_from_google()
    logger.info("Instructions synchronised from Google Sheets", extra={"count": synced_instructions})
except Exception:
    logger.exception("Failed to synchronise instructions from Google Sheets during startup")
    raise


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
        "СЃРїСЂР°РІРѕС‡РЅРёРє": "Catalog_",
        "СЃРїСЂР°РІРѕС‡РЅРёРєРё": "Catalog_",
        "catalog": "Catalog_",
        "catalogs": "Catalog_",
        "РєР°С‚Р°Р»РѕРі": "Catalog_",
        "РєР°С‚Р°Р»РѕРіРё": "Catalog_",
        # Documents
        "document": "Document_",
        "documents": "Document_",
        "РґРѕРєСѓРјРµРЅС‚": "Document_",
        "РґРѕРєСѓРјРµРЅС‚С‹": "Document_",
        "Р¶СѓСЂРЅР°Р»": "DocumentJournal_",
        "Р¶СѓСЂРЅР°Р»С‹": "DocumentJournal_",
        # Constants
        "constant": "Constant_",
        "constants": "Constant_",
        "РєРѕРЅСЃС‚Р°РЅС‚Р°": "Constant_",
        "РєРѕРЅСЃС‚Р°РЅС‚С‹": "Constant_",
        # Registers
        "РїР»Р°РЅ РѕР±РјРµРЅР°": "ExchangePlan_",
        "РїР»Р°РЅС‹ РѕР±РјРµРЅР°": "ExchangePlan_",
        "exchangeplan": "ExchangePlan_",
        "chart of accounts": "ChartOfAccounts_",
        "РїР»Р°РЅ СЃС‡РµС‚РѕРІ": "ChartOfAccounts_",
        "РїР»Р°РЅС‹ СЃС‡РµС‚РѕРІ": "ChartOfAccounts_",
        "chartofcalculationtypes": "ChartOfCalculationTypes_",
        "РїР»Р°РЅ РІРёРґРѕРІ СЂР°СЃС‡РµС‚Р°": "ChartOfCalculationTypes_",
        "РїР»Р°РЅС‹ РІРёРґРѕРІ СЂР°СЃС‡РµС‚Р°": "ChartOfCalculationTypes_",
        "chartofcharacteristictypes": "ChartOfCharacteristicTypes_",
        "РїР»Р°РЅ РІРёРґРѕРІ С…Р°СЂР°РєС‚РµСЂРёСЃС‚РёРє": "ChartOfCharacteristicTypes_",
        "СЂРµРіРёСЃС‚СЂ СЃРІРµРґРµРЅРёР№": "InformationRegister_",
        "СЂРµРіРёСЃС‚СЂС‹ СЃРІРµРґРµРЅРёР№": "InformationRegister_",
        "informationregister": "InformationRegister_",
        "СЂРµРіРёСЃС‚СЂ РЅР°РєРѕРїР»РµРЅРёСЏ": "AccumulationRegister_",
        "СЂРµРіРёСЃС‚СЂС‹ РЅР°РєРѕРїР»РµРЅРёСЏ": "AccumulationRegister_",
        "accumulationregister": "AccumulationRegister_",
        "СЂРµРіРёСЃС‚СЂ СЂР°СЃС‡РµС‚Р°": "CalculationRegister_",
        "СЂРµРіРёСЃС‚СЂС‹ СЂР°СЃС‡РµС‚Р°": "CalculationRegister_",
        "calculationregister": "CalculationRegister_",
        "СЂРµРіРёСЃС‚СЂ Р±СѓС…РіР°Р»С‚РµСЂРёРё": "AccountingRegister_",
        "СЂРµРіРёСЃС‚СЂС‹ Р±СѓС…РіР°Р»С‚РµСЂРёРё": "AccountingRegister_",
        "accountingregister": "AccountingRegister_",
        "Р±РёР·РЅРµСЃ РїСЂРѕС†РµСЃСЃ": "BusinessProcess_",
        "Р±РёР·РЅРµСЃ РїСЂРѕС†РµСЃСЃС‹": "BusinessProcess_",
        "businessprocess": "BusinessProcess_",
        "Р·Р°РґР°С‡Р°": "Task_",
        "Р·Р°РґР°С‡Рё": "Task_",
        "task": "Task_",
        "tasks": "Task_",
    }

    FIELD_SYNONYMS: Dict[str, List[str]] = {
        "РЅР°РёРјРµРЅРѕРІР°РЅРёРµ": ["Description", "РќР°РёРјРµРЅРѕРІР°РЅРёРµ", "Name"],
        "РёРјСЏ": ["Description", "Name", "РќР°РёРјРµРЅРѕРІР°РЅРёРµ"],
        "РѕРїРёСЃР°РЅРёРµ": ["Description", "РќР°РёРјРµРЅРѕРІР°РЅРёРµ"],
        "code": ["Code", "РљРѕРґ"],
        "РєРѕРґ": ["Code", "РљРѕРґ"],
        "Р°СЂС‚РёРєСѓР»": ["РђСЂС‚РёРєСѓР»", "SKU", "Code"],
        "РёРЅРЅ": ["РРќРќ", "Inn", "INN"],
        "РЅРѕРјРµСЂ": ["РќРѕРјРµСЂ", "Number", "РќРѕРјРµСЂР”РѕРєСѓРјРµРЅС‚Р°", "DocumentNumber"],
        "РёРґ": ["Ref_Key", "ID", "RefKey"],
        "guid": ["Ref_Key"],
        "РіРёРґ": ["Ref_Key"],
        "РєРѕР»РёС‡РµСЃС‚РІРѕ": ["РљРѕР»РёС‡РµСЃС‚РІРѕ", "Quantity"],
        "С†РµРЅР°": ["Р¦РµРЅР°", "Price"],
        "СЃСѓРјРјР°": ["РЎСѓРјРјР°", "Amount"],
        "СЃС‚РѕРёРјРѕСЃС‚СЊ": ["РЎСѓРјРјР°", "Amount", "Р¦РµРЅР°"],
        "РґР°С‚Р°": ["Р”Р°С‚Р°", "Date", "Р”Р°С‚Р°Р”РѕРєСѓРјРµРЅС‚Р°"],
        "РґР°С‚Р° РґРѕРєСѓРјРµРЅС‚Р°": ["Р”Р°С‚Р°", "Р”Р°С‚Р°Р”РѕРєСѓРјРµРЅС‚Р°", "Date"],
        "С„РѕСЂРјР°С‚": ["Р¤РѕСЂРјР°С‚", "Format"],
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
            ``РќР°РёРјРµРЅРѕРІР°РЅРёРµ``/``Name``) when no direct match is found.  When
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
            for default_field in ["Description", "РќР°РёРјРµРЅРѕРІР°РЅРёРµ", "Name"]:
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
        # prefer Description/РќР°РёРјРµРЅРѕРІР°РЅРёРµ/Name or first string property
        for cand in ["Description", "РќР°РёРјРµРЅРѕРІР°РЅРёРµ", "Name"]:
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
        """РЎС‚СЂРѕРёС‚ РІС‹СЂР°Р¶РµРЅРёРµ substringof СЃ РєРѕРЅСЃС‚Р°РЅС‚РѕР№ РІ РїРµСЂРІРѕРј Р°СЂРіСѓРјРµРЅС‚Рµ.

        OData РІ 1РЎ РѕР¶РёРґР°РµС‚, С‡С‚Рѕ РїРµСЂРІС‹Рј Р°СЂРіСѓРјРµРЅС‚РѕРј С„СѓРЅРєС†РёРё Р±СѓРґРµС‚ РїРѕРґСЃС‚СЂРѕРєР°,
        Р° РІС‚РѕСЂС‹Рј вЂ” РёРјСЏ РїРѕР»СЏ. РРјРµРЅРЅРѕ РїРѕСЌС‚РѕРјСѓ СЂР°РЅРµРµ РІРѕР·РЅРёРєР°Р»Р° РѕС€РёР±РєР°
        "РќРµРїСЂР°РІРёР»СЊРЅС‹Р№ С‚РёРї Р°СЂРіСѓРјРµРЅС‚Р°": РјС‹ РїРµСЂРµРґР°РІР°Р»Рё Р°СЂРіСѓРјРµРЅС‚С‹ РІ РѕР±СЂР°С‚РЅРѕРј
        РїРѕСЂСЏРґРєРµ. Р—РґРµСЃСЊ РјС‹ С„РѕСЂРјРёСЂСѓРµРј РєРѕСЂСЂРµРєС‚РЅРѕРµ РІС‹СЂР°Р¶РµРЅРёРµ
        substringof('<РїРѕРґСЃС‚СЂРѕРєР°>', Field).
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
        РЎРЅР°С‡Р°Р»Р° С‚РѕС‡РЅС‹Рµ eq РґР»СЏ РІСЃРµС… РїРѕР»РµР№.
        Р—Р°С‚РµРј РґР»СЏ РєР°Р¶РґРѕРіРѕ СЃС‚СЂРѕРєРѕРІРѕРіРѕ РїРѕР»СЏ С„РѕСЂРјРёСЂСѓРµРј РІР°СЂРёР°РЅС‚ СЃ substringof,
        РѕСЃС‚Р°Р»СЊРЅС‹Рµ РїРѕР»СЏ РѕСЃС‚Р°СЋС‚СЃСЏ РєР°Рє eq. РЎРѕР±РёСЂР°РµРј РІС‹СЂР°Р¶РµРЅРёРµ С‡РµСЂРµР· AND.
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
        РџСЂРѕРіСЂРµСЃСЃРёРІРЅС‹Р№ РїРѕРёСЃРє.

        - filters=str:
            * РµСЃР»Рё СЃС‚СЂРѕРєР° РїРѕС…РѕР¶Р° РЅР° РіРѕС‚РѕРІС‹Р№ OData `$filter`, РѕРЅ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ РєР°Рє РµСЃС‚СЊ;
            * РёРЅР°С‡Рµ РёС‰РµРј РїРѕ РѕСЃРЅРѕРІРЅРѕРјСѓ С‚РµРєСЃС‚РѕРІРѕРјСѓ РїРѕР»СЋ (eq в†’ substringof), Р·Р°С‚РµРј
              РІР°СЂРёР°РЅС‚С‹ СЃ `IsFolder eq false`, РµСЃР»Рё РµСЃС‚СЊ).
        - filters=dict: СЃРЅР°С‡Р°Р»Р° С‚РѕС‡РЅС‹Рµ `eq`, Р·Р°С‚РµРј РґР»СЏ СЃС‚СЂРѕРєРѕРІС‹С… РїРѕР»РµР№ РґРѕР±Р°РІР»СЏРµС‚СЃСЏ
          РІР°СЂРёР°РЅС‚ СЃ `substringof` (РѕСЃС‚Р°Р»СЊРЅС‹Рµ РѕСЃС‚Р°СЋС‚СЃСЏ `eq`), РїР»СЋСЃ РїРѕРїС‹С‚РєРё СЃ
          `IsFolder eq false`.

        Р’РѕР·РІСЂР°С‰Р°РµС‚ РїРµСЂРІС‹Р№ РЅРµРїСѓСЃС‚РѕР№ СЂРµР·СѓР»СЊС‚Р°С‚ (СѓС‡РёС‚С‹РІР°СЏ `top`).
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
                    # Р•СЃР»Рё top<=1 вЂ” РІРµСЂРЅСѓС‚СЊ РѕРґРёРЅРѕС‡РЅС‹Р№ РѕР±СЉРµРєС‚, РёРЅР°С‡Рµ СЃРїРёСЃРѕРє
                    if top is not None and int(top) <= 1:
                        res["data"] = vals[0]
                    else:
                        res["data"] = vals
                    return res
                if res.get("http_code") == 400 and res.get("odata_error_code") == 21:
                    continue
            # РµСЃР»Рё РІСЃРµ РїСѓСЃС‚Рѕ вЂ” РІРµСЂРЅС‘Рј РїРѕСЃР»РµРґРЅРёР№ res (РёР»Рё вЂњРїСѓСЃС‚РѕвЂќ)
            if attempt_list:
                return res
            # РІРѕРѕР±С‰Рµ Р±РµР· РїРѕРїС‹С‚РѕРє вЂ” РєР°Рє list_objects Р±РµР· С„РёР»СЊС‚СЂР°
            return self.list_objects(object_name, filters=None, top=top, expand=expand)

        # РЎР»РѕРІР°СЂРЅС‹Рµ С„РёР»СЊС‚СЂС‹: СЃРЅР°С‡Р°Р»Р° РЅРѕСЂРјР°Р»РёР·СѓРµРј РёРјРµРЅР° РїРѕР»РµР№
        if isinstance(user_filters, dict):
            normalized: Dict[str, Any] = {}
            for k, v in user_filters.items():
                field = self.resolve_field_name(object_name, k) or k
                normalized[field] = v
            # 1-СЏ РІРѕР»РЅР°: С‚РѕС‡РЅС‹Рµ eq
            attempts = [self._build_filter(normalized) or ""]
            # 2-СЏ РІРѕР»РЅР°: РґР»СЏ РєР°Р¶РґРѕРіРѕ СЃС‚СЂРѕРєРѕРІРѕРіРѕ РїРѕР»СЏ вЂ” substringof
            attempts += self._progressive_attempts_for_dict(object_name, normalized, include_only_elements=False)
            # РџСЂРѕР±СѓРµРј
            res = exec_attempts([a for a in attempts if a])
            if res.get("data"):
                return res
            # Р•СЃР»Рё РїСѓСЃС‚Рѕ вЂ” РґРѕР±Р°РІРёРј В«С‚РѕР»СЊРєРѕ СЌР»РµРјРµРЅС‚С‹В»
            if self._has_isfolder(object_name):
                attempts_is = self._progressive_attempts_for_dict(
                    object_name, normalized, include_only_elements=True
                )
                res = exec_attempts(attempts_is)
                return res

            return res  # РїСѓСЃС‚Рѕ

        # РЎС‚СЂРѕРєРѕРІС‹Р№ РїРѕРёСЃРє РёР»Рё РіРѕС‚РѕРІС‹Р№ С„РёР»СЊС‚СЂ
        if isinstance(user_filters, str) and user_filters:
            # Р•СЃР»Рё СЃС‚СЂРѕРєР° РїРѕС…РѕР¶Р° РЅР° РіРѕС‚РѕРІРѕРµ OData-РІС‹СЂР°Р¶РµРЅРёРµ вЂ” РёСЃРїРѕР»СЊР·СѓРµРј РµС‘ РЅР°РїСЂСЏРјСѓСЋ
            if self._looks_like_filter(user_filters):
                if top is not None and int(top) <= 1:
                    return self.find_object(object_name, filters=user_filters, expand=expand)
                return self.list_objects(object_name, filters=user_filters, top=top, expand=expand)

            # РРЅР°С‡Рµ СЃС‡РёС‚Р°РµРј, С‡С‚Рѕ СЌС‚Рѕ С‚РµРєСЃС‚ РґР»СЏ РїРѕРёСЃРєР° РїРѕ РѕСЃРЅРѕРІРЅРѕРјСѓ РїРѕР»СЋ
            fld = self._default_text_field(object_name)
            if not fld:
                # РљР°Рє fallback вЂ” РїСЂРѕСЃС‚Рѕ top(1) Р±РµР· С„РёР»СЊС‚СЂР°
                return self.find_object(object_name, filters=None, expand=expand)
            # 1-СЏ РІРѕР»РЅР°: eq в†’ substr в†’ substr_ci
            attempts = self._progressive_attempts_for_string(
                object_name, fld, user_filters, include_only_elements=False
            )
            res = exec_attempts(attempts)
            if res.get("data"):
                return res
            # 2-СЏ РІРѕР»РЅР°: РїРѕРІС‚РѕСЂ СЃ IsFolder eq false (РµСЃР»Рё РїРѕР»Рµ РµСЃС‚СЊ РІ СЃС…РµРјРµ)
            if self._has_isfolder(object_name):
                attempts2 = self._progressive_attempts_for_string(
                    object_name, fld, user_filters, include_only_elements=True
                )
                res = exec_attempts(attempts2)
                return res
            return res

        # РРЅР°С‡Рµ вЂ” РєР°Рє РѕР±С‹С‡РЅС‹Р№ РІС‹Р·РѕРІ
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
                "Entity name or phrase (e.g. Catalog_РљРѕРЅС‚СЂР°РіРµРЅС‚С‹ or "
                "'СЃРїСЂР°РІРѕС‡РЅРёРє РєРѕРЅС‚СЂР°РіРµРЅС‚С‹')"
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
    Р’РѕР·РІСЂР°С‰Р°РµС‚ РёРЅСЃС‚СЂСѓРєС†РёСЋ РёР· Р‘Р” РґР»СЏ СѓРєР°Р·Р°РЅРЅРѕР№ СЃСѓС‰РЅРѕСЃС‚Рё Рё РґРµР№СЃС‚РІРёСЏ.
    РРЅСЃС‚СЂСѓРєС†РёСЏ СЃРѕРґРµСЂР¶РёС‚ РѕРїРёСЃР°РЅРёРµ, РІРѕР·РјРѕР¶РЅС‹Рµ С€Р°РіРё, СЃС…РµРјСѓ Р°СЂРіСѓРјРµРЅС‚РѕРІ Рё
    РєР°СЂС‚Сѓ РїРѕР»РµР№. Р­С‚Рё РґР°РЅРЅС‹Рµ РЅРµРѕР±С…РѕРґРёРјРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РґР»СЏ РїРѕРґРіРѕС‚РѕРІРєРё С€Р°РіРѕРІ todo.

    РћР±СЂР°Р±Р°С‚С‹РІР°СЋС‚СЃСЏ СЃР»СѓС‡Р°Рё РѕС‚СЃСѓС‚СЃС‚РІРёСЏ РёРЅСЃС‚СЂСѓРєС†РёРё Рё РѕС‚СЃСѓС‚СЃС‚РІРёСЏ С€Р°РіРѕРІ.
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
                "РЎРѕРїРѕСЃС‚Р°РІСЊС‚Рµ user_args СЃ field_map Рё РІС‹Р·РѕРІРёС‚Рµ todo. Р•СЃР»Рё "
                "С€Р°РіРѕРІ РЅРµС‚ вЂ” РѕС‚РІРµС‚ СЃС„РѕСЂРјРёСЂСѓР№С‚Рµ РїРѕ descr."
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
        "http_message": f"РќРµРїРѕРґРґРµСЂР¶РёРІР°РµРјР°СЏ РѕРїРµСЂР°С†РёСЏ '{func}'",
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
                "РЁР°Рі РёР»Рё РјР°СЃСЃРёРІ С€Р°РіРѕРІ (func + args), РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅС‹С… РїРѕ "
                "РёРЅСЃС‚СЂСѓРєС†РёРё howto"
            )
        ),
    ] = None,
    instruction: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description=(
                "РћРїС†РёРѕРЅР°Р»СЊРЅРѕ: РѕР±СЉРµРєС‚ РёРЅСЃС‚СЂСѓРєС†РёРё РёР· howto. РСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ, "
                "РµСЃР»Рё С€Р°РіРё РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‚ вЂ” С‚РѕРіРґР° Р±СѓРґРµС‚ РІРѕР·РІСЂР°С‰С‘РЅ РѕС‚РІРµС‚ РЅР° "
                "РѕСЃРЅРѕРІРµ РѕРїРёСЃР°РЅРёСЏ (descr)."
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    Р’С‹РїРѕР»РЅСЏРµС‚ С€Р°РіРё OData: search|get|create|update|delete|post|unpost.
    Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃС‚Р°С‚СѓСЃС‹ РїРѕ С€Р°РіР°Рј Рё СЃРІРѕРґРєСѓ
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
                        "message": descr or "РќРµС‚ С€Р°РіРѕРІ.",
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
                    "error": f"РќРµРїРѕРґРґРµСЂР¶РёРІР°РµРјР°СЏ РѕРїРµСЂР°С†РёСЏ '{func}'",
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
        Field(description="РњР°С‚РµРјР°С‚РёС‡РµСЃРєРѕРµ РІС‹СЂР°Р¶РµРЅРёРµ РІ РІРёРґРµ СЃС‚СЂРѕРєРё",
        examples=["2+3", "4.2*5", "147/3", "56+68+72.3-10", "(19+134)**2", "log(10)", "sqrt(4)"],
        max_length=256)]
) -> Dict[str, Any]:
    """
    Р’РѕР·РІСЂР°С‰Р°РµС‚ СЂРµР·СѓР»СЊС‚Р°С‚ РІС‹С‡РёСЃР»РµРЅРёСЏ Р°СЂРёС„РјРµС‚РёС‡РµСЃРєРѕРіРѕ РІС‹СЂР°Р¶РµРЅРёСЏ.
    РџРѕРґРґРµСЂР¶РёРІР°СЋС‚СЃСЏ РѕРїРµСЂР°С†РёРё: +, -, *, /, //, **, sqrt, log, log10, exp
    РџРѕРґРґРµСЂР¶РёРІР°РµС‚СЃСЏ СѓРєР°Р·Р°РЅРёРµ РїРѕСЂСЏРґРєР° РѕРїРµСЂР°С†РёР№ СЃ РїРѕРјРѕС‰СЊСЋ СЃРєРѕР±РѕРє

    Args:
        expression: РјР°С‚РµРјР°С‚РёС‡РµСЃРєРѕРµ РІС‹СЂР°Р¶РµРЅРёРµ РІ РІРёРґРµ СЃС‚СЂРѕРєРё

    Returns:
        Dict СЃ СЃР»РµРґСѓСЋС‰РёРјРё РїРѕР»СЏРјРё:
        - success (bool): Р¤Р»Р°Рі СѓСЃРїРµС€РЅРѕРіРѕ РІС‹РїРѕР»РЅРµРЅРёСЏ РѕРїРµСЂР°С†РёРё
        - result (str|null): Р РµР·СѓР»СЊС‚Р°С‚ РІС‹С‡РёСЃР»РµРЅРёСЏ Р°СЂРёС„РјРµС‚РёС‡РµСЃРєРѕРіРѕ РІС‹СЂР°Р¶РµРЅРёСЏ РІ РІРёРґРµ СЃС‚СЂРѕРєРё
        - error (str|null): РЎРѕРѕР±С‰РµРЅРёРµ РѕР± РѕС€РёР±РєРµ РµСЃР»Рё РѕРїРµСЂР°С†РёСЏ РЅРµ СѓРґР°Р»Р°СЃСЊ

    РџСЂРёРјРµСЂС‹ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёСЏ:
       - РЈРјРЅРѕР¶РёС‚СЊ 147.5 РЅР° 5:
        solve("147.5*5")
       - Р’РѕР·РІРµСЃС‚Рё С‡РёСЃР»Рѕ 1440 РІ РєРІР°РґСЂР°С‚:
        solve("1440**2")
       - РџРѕСЃС‡РёС‚Р°С‚СЊ СЃСѓРјРјСѓ 1245, 1030, 5044, 986:
        solve("1245+1030+5044+986")
       - РџРѕСЃС‡РёС‚Р°С‚СЊ РґРµСЃСЏС‚РёС‡РЅС‹Р№ Р»РѕРіР°СЂРёС„Рј РѕС‚ 25:
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
                    raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјС‹Р№ СѓР·РµР»: {type(node).__name__}")
                if isinstance(node, (ast.BinOp)):
                    if not isinstance(node.op, allowed_operations):
                        raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјС‹Р№ Р±РёРЅР°СЂРЅС‹Р№ РѕРїРµСЂР°С‚РѕСЂ: {type(node.op).__name__}")
                if isinstance(node, (ast.UnaryOp)):
                    if not isinstance(node.op, allowed_operations):
                        raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјС‹Р№ СѓРЅР°СЂРЅС‹Р№ РѕРїРµСЂР°С‚РѕСЂ: {type(node.op).__name__}")
                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјРѕРµ РёРјСЏ: {node.id}")
                if isinstance(node, ast.Call):
                    if node.func.id not in allowed_names:
                        raise ValueError(f"РќРµРґРѕРїСѓСЃС‚РёРјР°СЏ С„СѓРЅРєС†РёСЏ: {node.func.id}")

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
                "error": f"РћС€РёР±РєР° СЂРµС€РµРЅРёСЏ РІС‹СЂР°Р¶РµРЅРёСЏ: {str(e)}"
            }

    result = await asyncio.to_thread(_sync)
    return _json_ready(result)


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import sys

    mcp.run(transport="sse")


