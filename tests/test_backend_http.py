"""Unit tests for the HTTP backend in Annif"""

import unittest.mock
import annif.backend.http


def test_http_analyze():
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = [
            {'uri': 'http://example.org/http', 'label': 'http', 'score': 1.0}]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend_type("http")
        http = http_type(
            config={'endpoint': 'http://api.example.org/analyze', 'project': 'dummy'})
        result = http.analyze('this is some text')
        assert len(result) == 1
        assert result[0].uri == 'http://example.org/http'
        assert result[0].label == 'http'
        assert result[0].score == 1.0