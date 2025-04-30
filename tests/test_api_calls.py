import unittest
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from utils.openai_query import openai_chat
from utils.anthropic_query import anthropic_chat

# Load environment variables for testing
load_dotenv()


# Test class for API calls
class TestAPICalls(unittest.TestCase):
    def setUp(self):
        """Set up test variables"""
        self.context = "You are an AI assistant specializing in gene analysis."
        self.prompt = "Analyze these genes: BRCA1, TP53, MLH1"
        self.test_log_file = "test_api.log"

        # Load config for testing
        with open("config.json", "r") as f:
            self.config = json.load(f)

    @patch("openai.chat.completions.create")
    def test_openai_call(self, mock_create):
        """Test OpenAI API call with mocked response"""
        # Set up the mock
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Test analysis"
        mock_resp.usage.total_tokens = 100
        mock_resp.system_fingerprint = "test-fingerprint"
        mock_create.return_value = mock_resp

        # Call the function
        model = "gpt-4"
        result, fingerprint = openai_chat(
            self.context,
            self.prompt,
            model,
            0.0,
            1000,
            0.00001,
            self.test_log_file,
            10.0,
            None,
        )

        # Check results
        self.assertEqual(result, "Test analysis")
        self.assertEqual(fingerprint, "test-fingerprint")
        mock_create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_anthropic_call(self, mock_anthropic):
        """Test Anthropic API call with mocked response"""
        # Set up the mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock()]
        mock_resp.content[0].text = "Test Claude analysis"
        mock_client.messages.create.return_value = mock_resp

        # Call the function
        model = "claude-3-7-sonnet-20250219"
        result, _ = anthropic_chat(
            self.context, self.prompt, model, 0.0, 1000, self.test_log_file, None
        )

        # Check results
        self.assertEqual(result, "Test Claude analysis")
        mock_client.messages.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
