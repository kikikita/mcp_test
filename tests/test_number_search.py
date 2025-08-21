from mcp_server import MCPServer


class DummyClient:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_metadata(self):
        return self._metadata


def make_server(metadata):
    srv = MCPServer.__new__(MCPServer)
    srv.client = DummyClient(metadata)
    return srv


def test_progressive_dict_numeric_field():
    metadata = {
        "Entity": {"properties": {"Number": {"type": "Edm.Int32"}}}
    }
    srv = make_server(metadata)
    attempts = srv._progressive_attempts_for_dict("Entity", {"Number": 123}, False)
    assert attempts == ["Number eq 123"]


def test_progressive_string_numeric_field():
    metadata = {
        "Entity": {"properties": {"Number": {"type": "Edm.Int32"}}}
    }
    srv = make_server(metadata)
    attempts = srv._progressive_attempts_for_string("Entity", "Number", "123", False)
    assert attempts == ["Number eq '123'"]
