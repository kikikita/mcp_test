from mcp_server import _unescape_strings


def test_unescape_strings_removes_backslashes():
    data = {"Description": "ООО \\'БАУМ ASTRA\\'"}
    cleaned = _unescape_strings(data)
    assert cleaned["Description"] == "ООО 'БАУМ ASTRA'"


def test_unescape_strings_keeps_other_backslashes():
    data = {"path": "C:\\Temp\\file"}
    cleaned = _unescape_strings(data)
    assert cleaned["path"] == "C:\\Temp\\file"
