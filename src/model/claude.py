
from src.utils.bedrock import Bedrock

class Claude:
    def __init__(self, region_name, credentials_profile_name=None, endpoint_url=None):
        self.bedrock = Bedrock(
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            model_id="anthropic.claude-v2:1",  # Ensure the correct model ID for Claude 2.1
            endpoint_url=endpoint_url,
            
        )

    def _call(self, prompt, stop=None, **kwargs):
        model_kwargs = {"max_tokens_to_sample": 128}  # Adjust as needed
        response = self.bedrock._call(prompt, stop=stop, **model_kwargs)
        return response

