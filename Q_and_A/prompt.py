GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["get_entity_questions"] = """

    I Will give you information about a subset of the entities in the dataset, and you use that information to generate questions about those entities or other relationships or entities that may be related
    Given the following entities description of a dataset:

    {entity_data}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 2 questions that was related to this dataset , it is esential that your questions can be answer with the knowledge gain in the dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """


PROMPTS["get_relation_questions"] = """

    I Will give you information about a subset of the relationships in the dataset, and you use that information to generate questions about those relationships or other relationships or entities that may be related.
    Given the following relationship description of a dataset:

    {relation_data}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 2 questions that was related to this dataset , it is esential that your questions can be answer with the knowledge gain in the dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """


PROMPTS["get_subgraph_questions"] = """

    I Will give you information about a subgraph in the dataset, it will contain entities and relationships, and you use that information to generate questions about this subgraph or related macroscopic problem, which needed full understanding of the dataset to answer.
    Given the following entities description of a dataset:

    {entity_data}

    And the following relationships description of a dataset:

    {relation_data}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 2 questions that was related to this dataset , it is esential that your questions can be answer with the knowledge gain in the dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """