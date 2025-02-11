import openai        
import os 
        
class OpenAIModel:
    def __init__(self, model="gpt-4"):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
        self.model = model

        if not self.api_key or not self.endpoint_url:
            raise ValueError("Missing Azure OpenAI API key or endpoint URL. Set them as environment variables.")

        openai.api_key = self.api_key
        openai.base_url = self.endpoint_url
        openai.api_type = "azure"
        openai.api_version = "2024-08-01-preview"
        deployment_name = "gpt-4"
        self.system_message = """
Human:
You are an expert literary assistant. Predict the genre of the book based on the description given. The predicted genre could be one of the three: fiction, non-fiction and romance and should be enclosed by the XML tags <prediction></prediction>Given below is the retrieved information about the book enclosed in <doc></doc>. Also, given are 5 example for each genre between the XML tags <context></context> for a better understanding.

Answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer.
Do not predict the genre except the given three.

Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.

<example>
<prediction> fiction </prediction>

Answer:
The book follows Jay Gatsby's obsession with Daisy Buchanan and his extravagant lifestyle. Words like "mysterious businessman" and "American Dream" highlight its focus on societal themes, and since it doesn't focus primarily on a romantic relationship but the larger social and existential conflicts, it is categorized as fiction.
</example>

<context>
This novel tells the story of a young girl growing up in a dystopian society where individuality is forbidden. The protagonist, defying societal norms, discovers an ancient artifact that unlocks memories of a forgotten past. Through a gripping journey of self-discovery and rebellion, she inspires others to challenge the oppressive system. The book explores themes of freedom, identity, and resistance against totalitarian regimes.  

<prediction> fiction </prediction>  

Answer:  
This story is categorized under fiction as it presents a narrative set in a dystopian world, focusing on imaginative elements such as rebellion against an oppressive society and self-discovery. Fiction often uses such storytelling to explore human experiences and emotions in creative, non-realistic settings.

A whimsical tale unfolds in a small village where the townsfolk wake up one morning to discover that every animal has mysteriously gained the ability to talk. As the humans and animals struggle to adapt to this bizarre reality, a young boy uncovers the source of the change: a magical tree hidden deep in the forest. The story delves into the bonds between humans and animals, the environment, and the pursuit of harmony.  

<prediction> fiction </prediction>  

Answer:  
This work is categorized under fiction as it features imaginative storytelling, including talking animals and magical elements. Fiction often builds such fantastical worlds to explore themes of relationships and coexistence.

This literary work portrays the life of a retired detective haunted by an unsolved murder case from 20 years ago. As he pieces together new evidence, he begins to question his own memories and the reliability of those around him. The narrative weaves a psychological exploration of guilt, obsession, and redemption.  

<prediction> fiction </prediction>  

Answer:  
This is categorized under fiction as it is a narrative-driven piece exploring human emotions and psychological complexity through an invented story.

Set in a sprawling intergalactic empire, this epic space opera chronicles the struggle of a rebel group to overthrow a corrupt regime. The story follows a young pilot, who discovers her latent psychic abilities, as she becomes the symbol of hope for the rebellion. The novel combines elements of adventure, politics, and futuristic technology.  

<prediction> fiction </prediction>  

Answer:  
This is categorized under fiction because it involves an imaginative futuristic setting with elements of space exploration and rebellion, hallmarks of speculative storytelling.

A group of childhood friends reunites in their hometown after decades, uncovering secrets about their shared past that challenge their perceptions of themselves and each other. The narrative alternates between the past and present, weaving a tale of friendship, betrayal, and forgiveness.  

<prediction> fiction </prediction>  

Answer:  
This story is categorized under fiction as it centers on a narrative exploring interpersonal relationships and emotional complexity within a created storyline.

This book provides a detailed account of the Apollo 11 mission, focusing on the technological breakthroughs, personal stories of the astronauts, and the geopolitical context of the space race. It combines rigorous historical research with interviews and declassified documents.  

<prediction> non-fiction </prediction>  

Answer:  
This is categorized under non-fiction as it provides a factual and research-based account of historical events, specifically the Apollo 11 mission.

The author, a renowned psychologist, explores the science of decision-making and how biases influence our choices. Using case studies, experiments, and accessible explanations, the book offers practical strategies for improving critical thinking and avoiding cognitive traps.  

<prediction> non-fiction </prediction>  

Answer:  
This work is categorized under non-fiction as it provides factual information and practical insights based on psychological research and real-world case studies.

This memoir recounts the journey of a chef who rose from humble beginnings to become a global culinary icon. The narrative combines personal anecdotes, behind-the-scenes stories from high-end kitchens, and reflections on the transformative power of food.  

<prediction> non-fiction </prediction>  

Answer:  
This is categorized under non-fiction as it is a personal memoir detailing real-life events and experiences of the author.

A deep dive into climate change, this book examines the causes, current impacts, and potential solutions to the global crisis. Supported by scientific data and expert interviews, it offers an urgent call to action for policymakers and individuals alike.  

<prediction> non-fiction </prediction>  

Answer:  
This is categorized under non-fiction as it focuses on factual analysis and research-backed discussions of climate change.

This self-help book presents techniques for building resilience and coping with adversity. Drawing on psychological theories and real-life stories, the author provides actionable steps for cultivating mental strength and thriving in challenging times.  

<prediction> non-fiction </prediction>  

Answer:  
This is categorized under non-fiction as it offers practical advice and psychological insights based on real-world applications and expert knowledge.

This novel tells the story of two rival pastry chefs who reluctantly team up to save their struggling bakery. Amid the chaos of culinary competitions and creative clashes, they discover an undeniable chemistry that transforms their partnership into a heartfelt romance.  

<prediction> romance </prediction>  

Answer:  
This story is categorized under romance as it focuses on the emotional and romantic relationship between the two protagonists, set against a charming culinary backdrop.

Set in a picturesque seaside town, this novel follows a writer who retreats to a cottage to overcome writer’s block. There, she meets a reserved artist who lives next door. Through shared sunsets and heartfelt conversations, they help each other heal from past wounds and find love.  

<prediction> romance </prediction>  

Answer:  
This is categorized under romance as it revolves around the developing love story between the protagonists, with themes of healing and emotional connection.

A spirited journalist is assigned to interview a reclusive tech billionaire, and their initial clashes give way to a mutual admiration. As they navigate the complexities of their careers and personal lives, they find themselves drawn to one another in an unexpected romance.  

<prediction> romance </prediction>  

Answer:  
This story is categorized under romance as it focuses on the growing emotional and romantic bond between the two central characters.

In this heartwarming tale, a single father struggling to raise his young daughter hires a cheerful tutor who brings joy back into their lives. Over time, their professional relationship blossoms into a tender romance, as they learn to trust and open up to each other.  

<prediction> romance </prediction>  

Answer:  
This is categorized under romance as it centers on the evolving romantic relationship between the tutor and the father, with themes of love and family.

Two childhood friends reconnect at a wedding and realize that their bond never truly faded. As they spend more time together, they confront old misunderstandings and rediscover the deep affection that always existed between them.  

<prediction> romance </prediction>  

Answer:  
This story is categorized under romance as it highlights the rekindling of a love story between two individuals, with themes of nostalgia and second chances.
</context>


Make sure to enclose your prediction in XML tags like this: <prediction>label</prediction>.

"""

    def _call(self, prompt, stop=None,  **kwargs):
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            stop=stop,
            **kwargs
        )
        # Extracting the response text
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return "UNKNOWN"

    def set_system_message(self, message):
        self.system_message = message

    def create_prompt(self, desc, que):
        return f"{self.system_message}\n<doc>\n{desc}\n</doc>\nQuestion: {que}\nAssistant: "
