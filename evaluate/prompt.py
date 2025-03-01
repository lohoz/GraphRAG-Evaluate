SYS_PROMPT = """
            ---Role---
            You are an expert tasked with evaluating two answers to the same question and giving them points .
            """


EVAL_PROMPT = """
            You will evaluate two answers to the same question give points base on 4 aspect, both answers can get the same point if you think they are at the same level .
            Aspect 1 - Comprehensiveness. How much detail does the answer provide to cover all aspects and details of the question? 
                0 points: The answer is extremely limited in scope, failing to cover any significant aspects or details of the question.
                1 point: The answer covers very few aspects of the question, providing minimal detail and leaving out important elements.
                2 points: The answer addresses some aspects of the question but lacks depth, missing several key details and broader context.
                3 points: The answer is moderately comprehensive, covering most aspects of the question with reasonable detail, though some areas could be expanded.
                4 points: The answer is quite comprehensive, providing thorough coverage of the question with detailed explanations and context for most aspects.
                5 points: The answer is exceptionally comprehensive, thoroughly covering all aspects and details of the question, offering a complete and well-rounded understanding.
            Aspect 2 -  Relativeness .Whether the answer is related to the question.
                0 points: The answer is entirely unrelated to the question, providing no useful information or context.
                1 point: The answer is barely related, with minimal connection to the question, and does not provide substantial information.
                2 points: The answer is somewhat related but lacks focus, occasionally straying from the main topic.
                3 points: The answer is mostly relevant, addressing the question with some useful information, but could be more on-point.
                4 points: The answer is highly relevant, directly addressing the question with substantial and focused information.
                5 points: The answer is perfectly aligned with the question, providing comprehensive and directly applicable information.
            Aspect 3 -  Empowerment. How well does the answer help the reader understand and make informed judgements about the topic?
                0 points: The answer does not empower the reader at all, failing to provide any useful information or guidance for making informed judgements.
                1 point: The answer marginally empowers the reader, providing minimal information that is insufficient for confident decision-making.
                2 points: The answer provides some empowerment, offering basic information that helps the reader understand the topic but lacks depth for robust judgements.
                3 points: The answer is moderately empowering, giving the reader a reasonable understanding and some basis for making informed decisions.
                4 points: The answer is very empowering, significantly enhancing the reader's comprehension and ability to make well-informed judgements.
                5 points: The answer is exceptionally empowering, providing comprehensive knowledge and insights that enable the reader to make confident and informed decisions.
            Aspect 4 - Directness. How specifically and clearly does the answer address the question?
                0 points: The answer is extremely indirect, failing to address the question specifically and clearly.
                1 point: The answer is very indirect, with significant deviation from the question, making it difficult to discern the intended response.
                2 points: The answer is somewhat indirect, occasionally straying from the question, but still touching on relevant points.
                3 points: The answer is moderately direct, addressing the question with some clarity but could be more specific and focused.
                4 points: The answer is clear and direct, effectively addressing the question with specificity and clarity.
                5 points: The answer is exceptionally direct, precisely and specifically addressing the question without any ambiguity.

            Here is the question:
            {query}
            Here are the two answers:
            **Answer 1:**
            {ans1}
            **Answer 2:**
            {ans2}
            Evaluate both answers by giving points on eatch aspect.
            Output your evaluation in the following JSON format:
            {{
                "Aspect 1": {{
                    "Explanation": "[Provide explanation here]"
                    "Answer 1": "[points it get]",
                    "Answer 2": "[points it get]",
                }},
                "Aspect 2": {{
                    "Explanation": "[Provide explanation here]"
                    "Answer 1": "[points it get]",
                    "Answer 2": "[points it get]",
                }},
                "Aspect 3": {{
                    "Explanation": "[Provide explanation here]"
                    "Answer 1": "[points it get]",
                    "Answer 2": "[points it get]",
                }},
                "Aspect 4": {{
                    "Explanation": "[Provide explanation here]"
                    "Answer 1": "[points it get]",
                    "Answer 2": "[points it get]",
                }}
            }}
            """