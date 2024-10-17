
from src.utils.bedrock import Bedrock

class Claude:
    def __init__(self, region_name, credentials_profile_name=None, endpoint_url=None):
        self.bedrock = Bedrock(
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            model_id="anthropic.claude-v2:1",  # Ensure the correct model ID for Claude 2.1
            endpoint_url=endpoint_url,
            
        )

        self.system_message = """
Human:
You are an expert literary assistant. Given below are example of the descriptions of different genres enclosed in <doc></doc> and following them are the answers in the format expected.

Answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer.

Consider the main genre and any possible sub-genre for the decription and include them in the prediction XML tag. 

Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
 
<doc>  
Harry Potter and the Philosopher's Stone is a fantasy novel written by J.K. Rowling. The story is set in a magical world and follows a young boy, Harry Potter, who discovers on his eleventh birthday that he is a wizard. He is whisked away to Hogwarts School of Witchcraft and Wizardry, where he begins his magical education and uncovers the truth about his parents' mysterious deaths and his own place in the wizarding world. Throughout the novel, Harry makes friends and encounters enemies, all while facing various magical challenges and uncovering the secrets of the Philosopher's Stone.
</doc> 

<example>
<prediction> fantasy </prediction>

Answer:  
Harry discovered he was a wizard on his eleventh birthday. He learned about his parents and his magical heritage from Hagrid. Since, we can observe the owrds like wizard and magic in the description. Also, author J. K Rowling is known for writing fantasy novels, we can conclde that the genre of the book "Harry Potter and the Philosophers Stone" is fantasy.
</example>

If the question cannot be answered by the book, say so.

Make sure to enclose your prediction in XML tags like this: <prediction>label</prediction>.

"""

    def _call(self, prompt, stop=None, **kwargs):
        model_kwargs = {"max_tokens_to_sample": 256}  # Adjust as needed
        response = self.bedrock._call(prompt, stop=stop, **model_kwargs)
        return response

    def set_system_message(self, message):
        self.system_message = message

    def create_prompt(self, desc, que):
        return f"{self.system_message}\n<doc>\n{desc}\n</doc>\nQuestion: {que}\nAssistant: "
