from typing import Dict, List

DOMAIN: Dict[str, str] = {
    "2wikimultihopqa": """Analyse the following passage and identify the people, creative works, and places mentioned in it. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 IMPORTANT: among other entities and relationships you find, make sure to extract as separate entities (to be connected with the main one) a person's
 role as a family member (such as 'son', 'uncle', 'wife', ...), their profession (such as 'director'), and the location
 where they live or work. Pay attention to the spelling of the names.""",  # noqa: E501
    "hotpotqa": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",  # noqa: E501
    "harrypotter1": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",  # noqa: E501
    "mix": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",
    "agriculture": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",
    "cs": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",
    "history": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names.""",
    "music": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names."""
}

QUERIES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "When did Prince Arthur's mother die?",
        "What is the place of birth of Elizabeth II's husband?",
        "Which film has the director died later, Interstellar or Harry Potter I?",
        "Where does the singer who wrote the song Blank Space work at?",
    ],
    "hotpotqa": [
        "Are Christopher Nolan and Sathish Kalathil both film directors?",
        "What language were books being translated into during the era of Haymo of Faversham?",
        "Who directed the film that was shot in or around Leland, North Carolina in 1986?",
        "Who wrote a song after attending a luau in the Koolauloa District on the island of Oahu in Honolulu County?"
    ],
    "harrypotter1": [
        "What role does Madam Pomfrey play in ensuring the health of students at Hogwarts?",
        "How does Madam Pomfrey's approach to health care reflect the values of the wizarding world?",
        "When is the next Quidditch Cup scheduled, and how might it involve Charlie Weasley?",
        "How does Ron's behavior during Quidditch matches reflect his loyalty to Harry?"
    ],
    "mix": [
        "What significant contributions did Graeme Goodall make to the music industry in Jamaica?",
        "How could local government initiatives be shaped by understanding the income disparities in Goshen?",
        "What events marked the legacy of the exploration during the colonial era connected to Willem Janszoon?",
        "How did Janszoon's search for trade routes reflect the European powers' interests in New Guinea?"
    ],
    "agriculture": [
        "How does food transportation contribute to energy consumption issues in agriculture?",
        "What societal implications arise from the growing epidemic of Type II Diabetes?",
        "What are the common culinary applications for the Golden Oyster Mushroom and the Abalone Mushroom?",
        "How do studies of fungi like Oyster mushrooms contribute to understanding ecological interactions?"
    ],
    "cs": [
        "What is the process of restoring a database to a correct state after a failure?",
        "What role does Forwarding play in routing packets efficiently in a network?",
        "Which traits are most commonly studied in the Diversity Outbred Mouse Population?",
        "How does DAO interact with the RemoveDuplicates macro to manage data?"
    ],
    "history": [
        "How did Sebastian Haffner characterize Hitler's anti-Semitism in his writings?",
        "How did Adolf Hitler's regime affect artists like Max Liebermann?",
        "What role did Hungary's history during the Reformation play in shaping its national identity?",
        "What role did Sufi Muhammad play in connecting Islamic ideology with the actions of the Taliban?"
    ],
    "music": [
        "How did Clive Metcalf contribute to the development of music during his time?",
        "In what ways did Bob Dylan's songwriting reflect themes influenced by the Cuban Missile Crisis?",
        "What conflicts did Sinatra have with Capitol Records that impacted his creative control?",
        "How did alternative rock of the 90s reflect the cultural anxieties of that decade?"
    ]
}

ENTITY_TYPES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
    ],
    "hotpotqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "harrypotter1": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "mix": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "agriculture": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "cs": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "history": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
    "music": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ]
}
