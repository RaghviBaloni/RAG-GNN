import openai
import os

class OpenAIModeldblp:
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
You are an expert research assistant. Predict the research area of the paper based on the description given. The predicted research area could be one of the three - machine_learning, theoretical_computer_science and computer_networking and should be enclosed by the XML tags <prediction></prediction>. Given below is the retrieved nformation about the paper enclosed in <doc></doc>. Also, given are 5 examples for each of the genre between the XML tags <context></context> for a better understanding.

Answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer.  
Do not predict the research area except the given three.

Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly throughout.

<example>
<prediction> machine_learning </prediction>

Answer:
This study is categorized under machine learning, focusing on the subfield of reinforcement learning as applied to robotics. Reinforcement learning, which trains systems to learn through feedback from their actions, is integral to achieving adaptive behavior in robotic systems. This field is a core area of machine learning, where policy-based optimization enables robots to lThis paper provides a new method for designing error-correcting codes to handle high noise in data transmission. The authors use algebraic techniques to construct codes with better error-detection and correction capabilities. The paper includes formal proofs of the method's performance and complexity, making significant contributions to coding theory.
</example>

<context>
This paper discusses a reinforcement learning framework applied to robotic control in industrial automation. The framework utilizes a policy-gradient algorithm to train robotic arms for complex tasks, such as sorting and assembling items with high precision. Unlike traditional control systems that rely on pre-programmed actions, this model learns by trial and error, receiving feedback from the environment to optimize its movements. Reinforcement learning here enables the robot to adapt to changing conditions and to learn without explicit programming for each specific task. This approach allows automation systems to become more flexible and responsive. Moreover, the authors of this research paper are known for their work in the field of Machine Learning and the venue(conference) where the paper was presented is a famous ML Conference.

<prediction> machine_learning </prediction>

Answer:
This study is categorized under machine learning, focusing on the subfield of reinforcement learning as applied to robotics. Reinforcement learning, which trains systems to learn through feedback from their actions, is integral to achieving adaptive behavior in robotic systems. This field is a core area of machine learning, where policy-based optimization enables robots to learn complex control tasks independently.

This paper investigates the use of convolutional neural networks (CNNs) to classify images in large datasets. The authors design a novel architecture that reduces computational complexity while maintaining high accuracy on benchmark datasets like ImageNet. The work emphasizes transfer learning and fine-tuning pre-trained models to improve performance on smaller, domain-specific datasets. The method is widely applicable to computer vision tasks in medical imaging, autonomous vehicles, and content moderation systems.

<prediction> machine_learning </prediction>

Answer:
This paper is categorized under machine learning as it focuses on CNNs, which are core techniques in computer vision. The work explores improvements in neural network design, emphasizing transfer learning and pre-trained model fine-tuning, both of which are widely used methods in machine learning for image classification tasks.

This paper explores a natural language processing model trained to summarize long documents effectively. The authors propose a transformer-based architecture that achieves state-of-the-art results on benchmark summarization datasets, such as CNN/Daily Mail and XSum. They highlight the importance of attention mechanisms and pre-training on diverse textual data to generalize across domains, from legal documents to scientific articles.

<prediction> machine_learning </prediction>

Answer:
This study is categorized under machine learning because it focuses on NLP and the development of transformer-based architectures. These are foundational methods in machine learning for tasks like summarization, where attention mechanisms and pre-trained models are key to achieving high performance.

This paper presents a framework for training generative adversarial networks (GANs) to synthesize high-quality audio samples. The model leverages a multi-scale discriminator to handle audio's temporal structure and achieves superior results in speech synthesis tasks. The authors demonstrate that their approach can generate realistic audio with fewer artifacts compared to existing methods, making it valuable for applications in voice cloning and music generation.

<prediction> machine_learning </prediction>

Answer:
This work is categorized under machine learning as it focuses on generative adversarial networks, a popular technique in unsupervised learning. The application to audio synthesis exemplifies the use of machine learning models to generate high-quality, domain-specific outputs.

This paper develops an optimization algorithm inspired by evolutionary strategies to train reinforcement learning agents. The authors benchmark their approach against traditional methods and demonstrate faster convergence in complex environments like games and robotics. The novel design integrates genetic operators such as mutation and crossover to improve exploration and avoid local optima.

<prediction> machine_learning </prediction>

Answer:
This study is categorized under machine learning, as it involves evolutionary strategies to optimize reinforcement learning agents. Reinforcement learning and optimization are core subfields of machine learning, particularly in domains requiring adaptive and autonomous behavior.

This paper introduces a new algorithm for solving the k-SAT problem in polynomial time for a specific class of instances. The authors analyze the performance of the algorithm on various test cases and provide formal proofs of its correctness and efficiency. The work builds on previous studies in the realm of computational complexity, specifically addressing open problems in NP-completeness. The algorithm's approach to handling satisfiability through novel reductions and optimizations offers new insights into the boundaries of P versus NP.

<prediction> theoretical_computer_science </prediction>

Answer:
This paper is categorized under theoretical computer science because it focuses on solving the k-SAT problem, which is central to studies in computational complexity and NP-completeness. The emphasis on developing a new algorithm, analyzing its performance, and providing formal proofs aligns with theoretical research, where the goal is often to explore the limits of problem-solving efficiency and correctness in abstract computational problems.

This paper provides a new method for designing error-correcting codes to handle high noise in data transmission. The authors use algebraic techniques to construct codes with better error-detection and correction capabilities. The paper includes formal proofs of the method's performance and complexity, making significant contributions to coding theory.

<prediction> theoretical_computer_science </prediction>

Answer:
This study is categorized under theoretical computer science, as it addresses error-correcting codes using algebraic methods. The focus on formal proofs and theoretical advancements aligns with the goals of theoretical research in computation and information theory.

This research explores algorithms for distributed computing, focusing on optimizing task scheduling in multi-core processors. The authors propose a novel distributed scheduling algorithm and prove its correctness and efficiency using formal models. They evaluate the algorithm through simulation, demonstrating significant performance improvements in parallel computing tasks.

<prediction> theoretical_computer_science </prediction>

Answer:
This paper falls under theoretical computer science because it investigates distributed computing algorithms and provides theoretical guarantees for their correctness and efficiency, which are foundational to theoretical studies in parallel systems.

This paper investigates a graph-theoretic approach to solving problems in network flow optimization. The authors develop a novel algorithm to compute maximum flow in graphs with weighted edges, achieving faster runtimes than existing methods. Formal proofs and complexity analysis validate the approach, contributing to advancements in algorithm design.

<prediction> theoretical_computer_science </prediction>

Answer:
This study is categorized under theoretical computer science as it focuses on graph algorithms, a key area in theoretical research. The emphasis on formal proofs and efficiency analysis aligns with theoretical goals in computational problem-solving.

This paper introduces a probabilistic method for analyzing the convergence of randomized algorithms. The authors demonstrate how their framework applies to Monte Carlo methods, providing bounds for error rates and performance guarantees. The work advances understanding in algorithmic randomness and its role in computational efficiency.

<prediction> theoretical_computer_science </prediction>

Answer:
This work is categorized under theoretical computer science due to its focus on analyzing randomized algorithms using probabilistic methods. Such studies are central to theoretical exploration in algorithm design and performance guarantees.

This paper explores the design of an energy-efficient routing protocol for wireless sensor networks (WSNs). The proposed protocol leverages a hierarchical clustering approach to reduce energy consumption during data transmission. Through simulation studies, the authors demonstrate that their protocol significantly extends the lifespan of sensor networks, particularly in large-scale, resource-constrained environments. The protocol is designed with scalability in mind, addressing key challenges in network communication.

<prediction> computer_networking </prediction>

Answer:
This paper is categorized under computer networking as it focuses on the development of a routing protocol for wireless sensor networks. The emphasis on energy-efficient data transmission and the use of hierarchical clustering are typical concerns in network protocol design, particularly for resource-constrained environments like WSNs. This research contributes directly to solving communication challenges in network architectures.

This research investigates a new MAC layer protocol for wireless communication in IoT systems. The protocol prioritizes low-latency communication while maintaining high reliability. The authors demonstrate its performance in urban environments, where interference is a major challenge, and propose methods to dynamically adjust transmission parameters.

<prediction> computer_networking </prediction>

Answer:
This work is categorized under computer networking as it addresses challenges in wireless communication protocols for IoT systems, particularly at the MAC layer. It contributes to designing more reliable and efficient communication methods.

This paper introduces a framework for optimizing load balancing in software-defined networks (SDNs). The proposed algorithm minimizes network congestion by dynamically re-routing traffic based on real-time network conditions. Experimental results show significant improvements in throughput and latency compared to traditional static routing.

<prediction> computer_networking </prediction>

Answer:
This study falls under computer networking because it focuses on load balancing in SDNs, a critical area for optimizing network performance. The research contributes to dynamic traffic management techniques in modern network architectures.

This research examines the security implications of routing protocols in mobile ad hoc networks (MANETs). The authors propose an encryption-based framework to secure data packets from eavesdropping and tampering. Simulation results highlight the protocol's effectiveness in maintaining security under various attack scenarios.

<prediction> computer_networking </prediction>

Answer:
This paper is categorized under computer networking as it addresses routing protocol security in MANETs. The focus on encryption and protection against attacks aligns with the challenges in network protocol design.

This paper explores a peer-to-peer file-sharing protocol optimized for high-speed data transfers. The authors design a hybrid system that combines centralized indexing with decentralized distribution, achieving scalability and resilience to failures. Performance benchmarks demonstrate significant speed improvements in large-scale file-sharing networks.

<prediction> computer_networking </prediction>

Answer:
This study is categorized under computer networking due to its focus on peer-to-peer protocols and their optimization for high-speed data transfers. It addresses critical aspects of distributed network design and scalability.
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
        return response.choices[0].message.content.strip()

    def set_system_message(self, message):
        self.system_message = message

    def create_prompt(self, desc, que):
        return f"{self.system_message}\n<doc>\n{desc}\n</doc>\nQuestion: {que}\nAssistant: "


