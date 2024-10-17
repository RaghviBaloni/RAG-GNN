
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
You are an expert research assistant. Prediction should be one of the research areas - machine_learning, theoretical_computer_science, computer_networking. Do not predict any research area out of the 3 given labels.

In the given subgraphs for each paper, node_id are the ids assigned to each node attribute such as author name, venue(Conference), abstract, etc. And the src, edge_attr and dst signify the relation between the src node_id(paper id) and the dest node_id. With the help of the subgraph, given a detailed explanation for your prediction for each paper's research area.

Here is a subgraph description of the paper you will answer questions about for your reference:  

<doc>
node_attr,node_id
Blockchain applications in healthcare,0
Alice Smith,1
XYZ University,2
Cryptographic techniques,3
Decentralized healthcare system,4
Paper ID: 001,5
,6
Research in secure data transfer,7
"Utilizes cryptographic techniques to ensure data integrity and confidentiality. Proposes a decentralized system for sharing medical records between institutions and highlights improvements over centralized systems.",8

src,edge_attr,dst
0,n_citation,5
0,abstract,8
0,author_org,2
0,author_name,1
3,tech_used_in,4
1,affiliated_with,2
0,tech_used,3
2,research_focus,7
</doc>  

Answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer.  

Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.  

<example>
<prediction> data_security_and_privacy </prediction>

Answer:
This paper is categorized under data security and privacy as it discusses using blockchain technology to enhance privacy and secure the transfer of medical records in healthcare. The authors propose a decentralized system employing cryptographic techniques to protect patient data. Given the emphasis on security improvements over traditional centralized systems, the research clearly falls within the domain of data security and privacy.

"""

    def _call(self, prompt, stop=None, **kwargs):
        model_kwargs = {"max_tokens_to_sample": 256}  # Adjust as needed
        response = self.bedrock._call(prompt, stop=stop, **model_kwargs)
        return response

    def set_system_message(self, message):
        self.system_message = message

    def create_prompt(self, desc, que):
        return f"{self.system_message}\n<doc>\n{desc}\n</doc>\nQuestion: {que}\nAssistant: "


