import logging

import mcp_server
from mcp_server import MCPServer


class DummyResponse:
    def values(self):
        return [{"Ref_Key": "1"}]


class DummyBuilder:
    def __init__(self, client):
        self.client = client
        self.expanded = None

    def expand(self, fields):
        self.expanded = fields
        return self

    def top(self, count):
        return self

    def filter(self, flt):
        return self

    def get(self):
        return DummyResponse()

    def create(self, data):
        return DummyResponse()

    def update(self, data=None):
        return DummyResponse()


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.http_code = 200
        self.http_message = "OK"
        self.odata_code = None
        self.odata_message = None
        self.last_id = None
        self.last_builder = None

    def get_metadata(self):
        return {
            "Catalog_Номенклатура": {
                "entity_type": "t",
                "properties": {},
                "navigation_properties": {"Parent": {}, "ЕдиницаИзмерения": {}}
            }
        }

    def __getattr__(self, item):
        self.last_builder = DummyBuilder(self)
        return self.last_builder

    def get_http_code(self):
        return self.http_code

    def get_http_message(self):
        return self.http_message

    def get_error_code(self):
        return self.odata_code

    def get_error_message(self):
        return self.odata_message

    def get_last_id(self):
        return self.last_id


def create_server(monkeypatch):
    monkeypatch.setattr(mcp_server, "ODataClient", DummyClient)
    return MCPServer("http://example.com")


def test_invalid_expand_segment_skipped(monkeypatch, caplog):
    server = create_server(monkeypatch)
    with caplog.at_level(logging.WARNING):
        server.list_objects("Catalog_Номенклатура", expand="Unknown")
    assert server.client.last_builder.expanded is None
    assert "Navigation property" in caplog.text


def test_mix_expand_segments(monkeypatch, caplog):
    server = create_server(monkeypatch)
    with caplog.at_level(logging.WARNING):
        server.list_objects("Catalog_Номенклатура", expand="Parent,Unknown")
    assert server.client.last_builder.expanded == "Parent"
    assert "Unknown" in caplog.text
