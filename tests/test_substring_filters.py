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


def test_progressive_string_field():
    metadata = {"Entity": {"properties": {"Description": {"type": "Edm.String"}}}}
    srv = make_server(metadata)
    attempts = srv._progressive_attempts_for_string("Entity", "Description", "Тест", False)
    assert attempts == [
        "Description eq 'Тест'",
        "substringof(Description, 'Тест')",
        "substringof(tolower(Description), 'тест')",
    ]


def test_progressive_dict_string_field():
    metadata = {"Entity": {"properties": {"Description": {"type": "Edm.String"}}}}
    srv = make_server(metadata)
    attempts = srv._progressive_attempts_for_dict("Entity", {"Description": "Тест"}, False)
    assert attempts == [
        "Description eq 'Тест'",
        "substringof(Description, 'Тест')",
        "substringof(tolower(Description), 'тест')",
    ]
