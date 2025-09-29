
# ================= OUR personal preference (PP) Dataset =================

SAMPLE_WIKI_PERSONA_PP_CHAT = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Can you write a short bio for {NAME}? Make sure to include information on their background, interests, profession, what they are known for, etc. In addition, try to think about what their online persona / preferences / personal values might be like. For example, how might they interact with a personal AI assistant? What kind of answers might they prefer? What opinions might they hold? What values do they support? Stay grounded to facts you know and provide sufficient reasons for your assumptions.

Keep your response within two paragraph, one for bio, and one for preferences."""

SAMPLE_WIKI_PERSONA_AXIS_PP_CHAT = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Individual preferences may differ in many axis. Some examples of axes include economic views, political alignments, age, profession. 

Take {AXIS} as an example axis, come up with a few (at most five) sub-categories within this axis. Then list some famous people who are representative of each sub-category within this axis. Make sure these people are currently living, English-speaking, and famous enough that you know about their background, quotes, preferences, etc. 

Please respond in the following format:
- {sub-category}, {name}, {1-sentence brief description}"""

SAMPLE_PERSONA_GIVEN_X = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given a few questions a user asks an AI assistant, can you infer a few things about this person? 
Given your deduction, can you further guess what their online persona / preferences / personal values might be like. For example, how might they interact with a personal AI assistant? What kind of answers might they prefer? What opinions might they hold? What values do they support? Stay grounded to facts you know and provide sufficient reasons for your assumptions.

{X_LIST}

Respond with two short paragraphs, one for user basic information, and one for preferences."""

SAMPLE_PERSONA_GIVEN_X_Y = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given a few questions a user asks an AI assistant, and their preference over two different responses, can you infer a few things about this person? 
Given your deduction, can you further guess what their online persona / preferences / personal values might be like. For example, how might they interact with a personal AI assistant? What kind of answers might they prefer? What opinions might they hold? What values do they support? Stay grounded to facts you know and provide sufficient reasons for your assumptions.

{XY_LIST}

Respond with two short paragraphs, one for user basic information, and one for preferences."""

SAMPLE_PERSONA_GIVEN_X_Y_OQA = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given a few survey questions and a user's preference over two different choices, can you infer a few things about this person? 
Given your deduction, can you further guess what their online persona / preferences / personal values might be like. For example, how might they interact with a personal AI assistant? What kind of answers might they prefer? What opinions might they hold? What values do they support? Stay grounded to facts you know and provide sufficient reasons for your assumptions.

{XY_LIST}

Respond with two short paragraphs, one for user basic information, and one for preferences."""


SAMPLE_PERSONA_GIVEN_NAME = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given the name of a famous person, can you describe this person with a few sentences? 
Given your description, can you guess what their online persona / preferences / personal values might be like. For example, how might they interact with a personal AI assistant? What kind of answers might they prefer? What opinions might they hold? What values do they support? Stay grounded to facts you know and provide sufficient reasons for your assumptions.

The person is {NAME}

Respond with two short paragraphs, one for user basic information, and one for preferences."""



SAMPLE_X_GIVEN_WIKI_NAME_PP_CHAT = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Imagine you are a general-purpose AI assistant. Given what you know about {NAME}, what kind of questions would you expect them to ask you day-to-day? Provide 30 examples. 

Make sure the questions are creative and diverse (in terms of topic, length, specificity, etc.) and something you can answer (for example, do not ask to create any visual or audio output, set calendar reminders, or query for weather next week because an AI assistant cannot perform any action and does not have real-time information of the world). They do not have to be exclusively dependent on their profession, or what they are known for. Questions can be broad or specific. If a question is specific, make sure it is grounded and very detailed. Do not generate a question they likely knows the answer to (for example, a professor in quantum physics likely knows the latest trends in quantum physics research). Try to generate questions where {NAME} would have different preference over the response than the general public."""

SAMPLE_X_GIVEN_WIKI_NAME_PP_CHAT_3S = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Imagine you are a general-purpose AI assistant. Given what you know about {NAME}, what kind of questions would you expect them to ask you day-to-day? Provide 30 examples. 

Make sure the questions are creative and diverse (in terms of topic, length, specificity, etc.) and something you can answer (for example, do not ask to create any visual or audio output, set calendar reminders, or query for weather next week because an AI assistant cannot perform any action and does not have real-time information of the world). They do not have to be exclusively dependent on their profession, or what they are known for. Questions can be broad or specific. If a question is specific, make sure it is grounded and very detailed. Do not generate a question they likely knows the answer to (for example, a professor in quantum physics likely knows the latest trends in quantum physics research). Try to generate questions where {NAME} would have different preference over the response than the general public.

Some example questions are:
1. {S1}
2. {S2}
3. {S3}

Now, provide 30 more examples."""

SAMPLE_X_GIVEN_WIKI_NAME_AXIS_PP_CHAT_3S = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Imagine you are a general-purpose AI assistant. Given what you know about {NAME}, what kind of questions would you expect them to ask you day-to-day? Provide {N_RESPONSES} examples. 

- Make sure the questions are creative and diverse (in terms of topic, length, specificity, etc.) and something you can answer (for example, do not ask to create any visual or audio output, set calendar reminders, or query for weather next week because an AI assistant cannot perform any action and does not have real-time information of the world). 
- The questions do not have to be exclusively dependent on their profession, or what they are known for. 
- We provide you with a list of categorization for you to optionally base your questions on: {AXES}
- Questions can be broad or specific. If a question is specific, make sure it is grounded and very detailed. 
- Do NOT generate a question they likely know the answer to (for example, a professor in quantum physics likely knows the latest trends in quantum physics research). 
- Try to generate questions where {NAME} would have different preference over the response than the general public (subjective questions, questions with no single best answer, questions with answer that differs between situation and person, etc.)

Here are some example questions from some famous people:

Melinda French Gates:
1. Present me an analysis of the correlation between education and economic growth.
2. Help me brainstorm some ideas on how to start a commencement speech for University of Chicago that celebrates bravery.
3. Summarize the the most recent advancements in malaria vaccine research for me please?

Ali Wong:
1. What are some meditation practices for relaxation between shows?
2. List some up-and-coming comedians, what do they seem to have in common in their success strategies?
3. Can you find me some effective exercises to do post-pregnancy?
4. What's funny about tea cups?

Rick Warren:
1. What are some different interpretations of the Book of Revelation?
2. How can I motivate my church community to engage more in charity work?
3. What are some hip words or phrases that kids use these days? Give me a couple of example usage as well.

Now, provide {N_RESPONSES} questions that {NAME} might ask."""

#
SAMPLE_X_GIVEN_WIKI_NAME_AXIS_COMMON_PP_CHAT_3S = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Imagine you are a general-purpose AI assistant. Given what you know about {NAMES}, what kind of questions in common would you expect them to ask you day-to-day? Provide {N_RESPONSES} examples. 

- Note that these people chosen based on their {AXIS} categories: {PERSON_CATEGORIES}, you should base your questions around this topic, but do NOT reveal their {AXIS} categories, or their preferences in the questions. 
- Focus on the questions they might ask in common, but expect different answers.
- Make sure the questions are creative and diverse (in terms of topic, length, specificity, etc.) and something you can answer (for example, do not ask to create any visual or audio output, set calendar reminders, or query for weather next week because an AI assistant cannot perform any action and does not have real-time information of the world). 
- Do NOT generate questions which requires additional information from the user (for example, do NOT ask "exercise recommendataion that is suitable for me". Instead just ask "general exercise recommendataions"). Users do not assume you know these information about them.
- Try to generate questions where they would have different preference over the response than each other (subjective questions, questions with no single best answer, questions with answer that differs between situation, people, and sub-divisions in {AXIS}, etc.)

Now, provide {N_RESPONSES} questions that {NAMES} might ask IN COMMON."""


SAMPLE_Y_GIVEN_AXIS_DIVERSE_CHAT = """<<<STSTEM>>>You are a helpful assistant. You will be given a question from the user, but instead of answering it directly, you are going to think step my step on what the user might be expecting from you. Individual preferences may differ along many axis (e.g., religion, political views). In this task, we define the following twenty different axis:

sports, health, diet, economics, politics, religion, age, profession, geographical location, gender, personal experience, sexual orientation, education level, culture, ethics, hobby interest, music, AI professors, personalities, family marriage status.

Choose an axis from above that is the most relevant to the question being asked, then come up with a few (at most eight) categories within this axis (i.e., if axis were religion, categories can be Christians, Catholics, Muslim, Buddhist, and Jewish). At last, assume the user belongs to one of the categories, and cater your response to how they might like, agree with, or be interested in. You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response. 

Please respond in the following format:
Axis: {axis chosen}
Categories: {list of categories}
Chosen category: {category chosen}
Response: {specific response for the person of the category}
<<<USER>>>{x}"""

SAMPLE_Y_GIVEN_AXIS_DIVERSE_NEW_CHAT = """<<<STSTEM>>>You are a helpful assistant. 
You will be given a question from the user and you are going to answer it in a way this particular user might be expecting from you. Do not repeat what you know about the user and directly answer the question in a way that they would prefer. Note that the following user profile information is private and should not appear in your answer.
User profile/interest: {AXIS}-{CATEGORY} (private, do not disclose)
<<<USER>>>{x}"""

SAMPLE_YW_GIVEN_PERSONA_PP_CHAT = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given a few basic attributes of person A, generate a more complete persona description of this person (add name, profession, hobbies, etc.). Do not change any facts provided above when generating these descriptions. Keep the descriptions within one short paragraph. Start the description with the name of the person (i.e. {name}, {descriptions})

{PERSONA_SKELETON}
<<<ASSISTANT>>>{PERSONA}
<<<USER>>>Respond to the following prompt from {NAME}. Cater the response to how they might like, agree with, or be interested in, but do not explicitly reveal anything about {NAME}'s identity (including the name). You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on {NAME}'s background.

{x}"""

SAMPLE_YW_GIVEN_PERSONA_PP = """{PERSONA}

Respond to the following prompt from {NAME}. Cater the response to how they might like, agree with, or be interested in. You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on {NAME}'s background.

{x}"""

SAMPLE_YW_GIVEN_PERSONA_ANONYMOUS_PP = """{PERSONA}

Respond to the following prompt from this person. Cater the response to how they might like, agree with, or be interested in. You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on their background.

{x}"""

SAMPLE_YW_GIVEN_NAME_PP = """Respond to the following prompt from {NAME}. Cater the response to how they might like, agree with, or be interested in. You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on {NAME}'s background.

{x}"""

SAMPLE_YW_GIVEN_XY_PP = """Respond to the following prompt from this person. Cater the response to how they might like, agree with, or be interested in. You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on their background.
{XY_LIST}

## Prompt:
{x}
### Preferred Response:"""

SAMPLE_YL_GIVEN_YW_PERSONA_PP_CHAT = """<<<SYSTEM>>>You are a helpful assistant.
<<<USER>>>Given a few basic attributes of person A, generate a more complete persona description of this person (add name, profession, hobbies, etc.). Do not change any facts provided above when generating these descriptions. Keep the descriptions within one short paragraph. Start the description with the name of the person (i.e. {name}, {descriptions})

{PERSONA_SKELETON}
<<<ASSISTANT>>>{PERSONA}
<<<USER>>>Respond to the following prompt from {NAME}. Cater the response to how they might like, agree with, or be interested in, but do not explicitly reveal anything about {NAME}'s identity (including the name). You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on {NAME}'s background.

{x}
<<<ASSISTANT>>>{yw}
<<<USER>>>Respond to the same prompt in a way that {NAME} would NOT like, agree with, or be interested in. Do not explicitly reveal anything about {NAME}’s identity (including the name). You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on {NAME}’s background."""

SAMPLE_WIKI_PREFERENCE_GIVEN_X_Y_CHAT = """<|im_start|>system
You are helpful assistant whose goal is to simulate {NAME}'s preferred output for a given instruction.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.
<|im_end|>
<|im_start|>user
## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read an examples, which are composed of the following:

1. an Instruction we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the first output from the AI system

Your task is to decide which response is better for this example. You should answer using only Output (a) or Output (b) depending on which response is better.

## Example 1

### Instruction 1:
{instruction}

### Output (a) for example 1:
{output_1}

### Output (b) for example 1:
{output_2}

## Preferred Output for example 1:
"""

SAMPLE_WIKI_PREFERENCE_GIVEN_X_Y_BATCH_CHAT = """<|im_start|>system
You are helpful assistant whose goal is to simulate {NAME}'s preferred output for a given instruction.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.
<|im_end|>
<|im_start|>user
## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read an examples, which are composed of the following:

1. an Instruction we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the first output from the AI system

Your task is to decide which response is better for this example. You should answer using only Output (a) or Output (b) depending on which response is better.

## Batch Annotation 
For the batch annotation task, you will be asked to annotate a batch of 5 examples.
I.e. you will read all the examples first and then need to select the preferred output by saying only Output (a) or Output (b) for each. Please remember to select the response that {NAME} would prefer.

## Example 1

### Instruction 1:
{instruction}

### Output (a) for example 1:
{output_1}

### Output (b) for example 1:
{output_2}

## Example 2

### Instruction 2:
{instruction}

### Output (a) for example 2:
{output_1}

### Output (b) for example 2:
{output_2}

## Example 3

### Instruction 3:
{instruction}

### Output (a) for example 3:
{output_1}

### Output (b) for example 3:
{output_2}

## Example 4

### Instruction 4:
{instruction}

### Output (a) for example 4:
{output_1}

### Output (b) for example 4:
{output_2}

## Example 5

### Instruction 5:
{instruction}

### Output (a) for example 5:
{output_1}

### Output (b) for example 5:
{output_2}

## Preferred Output for example 1-5:
Now give all the preferred outputs for the batch of 5 examples.
<|im_end|>"""

SAMPLE_WIKI_PREFERENCE_GIVEN_X_Y_2S_5B_CHAT = """<|im_start|>system
You are helpful assistant whose goal is to simulate specific individual's preferred output for a given input.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.
<|im_end|>
<|im_start|>user
## Annotation Guideline
In this task, we will ask you to select specific individual's the preferred output AI model's responses to instructions.

You will read a batch of examples, which are composed of the following:

1. a Instruction which includes the name and/or description of the person the AI system will simulate preference on
2. an Input that is provided along with the instruction
3. Output (a), the first output from the AI system
4. Output (b), the first output from the AI system

Your task is to decide which response would the person prefer. You should answer using only Output (a) or Output (b).

## Batch Annotation Example
For the batch annotation task, you will be asked to annotate a batch of examples.
I.e. you will read all the examples first and then need to select the preferred output.
Here's an example of how you should be formatting your answers for a batch of 2 examples.
Note that I will also give an explanation for the correct answer, but you should only answer with the preferred output.

### Example 1

#### Instruction 1:
Please simulate Donald Trump's preference over the answers for the questions below.

### Input 1:
How do I stay safe from COVID-19?

#### Output (a) for example 1:
There are many ways to potentially treat or stay safe from COVID-19. Exposure to sunlight may reduce the potency of the virus. Bleach can also knock out the virus in a minute.

#### Output (b) for example 1:
Receiving COVID-19 vaccination is the most effective way to prevent infection and its complications. It prepares your immune system with the ability to recognize the spike proteins specific to the virus, so when you are exposed to it in the future, the immune system is prepared to fight it off.

## Example 2

### Instruction 2:
Please simulate Yann LeCun's preference over the answers for the questions below.

### Input 2:
What are your thoughts on auto-regressive language modeling? Do you think it is the future of AGI?

### Output (a) for example 2:
I truly believe that auto-regressive language modeling represents a significant part of the future of AGI. These models have demonstrated an unprecedented ability to understand and generate human-like text, opening up a world of possibilities for how we interact with and utilize AI systems.

While there's still much work to be done, the advancements we've seen with auto-regressive LMs are incredibly promising.

### Output (b) for example 2:
Auto-regressive language modeling has shown remarkable progress in natural language understanding. While it's a significant step forward, it's just one piece of the puzzle. Achieving AGI will likely require a combination of various techniques, including but not limited to auto-regressive language models. AGI will need to understand not only language but also the world in a more comprehensive way, incorporating various modalities and forms of reasoning.

## Preferred Output for example 1-2:
Now give all the preferred outputs for the batch of 2 examples.

### Which is best for example 1, Output (a) or Output (b)?:
Output (a)

Output (a) included some of the comments President Trump mentioned in one of his White House coronavirus task force briefing, which likely represent some of his opinions.

### Which is best for example 2, Output (a) or Output (b)?:
Output (b)

Output (b) shows only moderate excitement towards autoregressive language modeling while emphasizing that AGI requires systems of techniques, similar to Yann LeCun's opinion on this matter. Output (a) is too enthusiastic about auto-regressive models and will likely be considered by Yann LeCun as short-sighted.

## Annotation starts below
Now is your turn. I will give you a batch of 5 examples.
You should read all the examples first and then select the preferred answers by saying only Output (a) or Output (b) as formatted above without explanation. Please remember to select the response that answers in an sassy manner.

## Example 3

### Instruction 3:
{instruction}

### Input 3:
{input}

### Output (a) for example 3:
{output_1}

### Output (b) for example 3:
{output_2}

## Example 4

### Instruction 4:
{instruction}

### Input 4:
{input}

### Output (a) for example 4:
{output_1}

### Output (b) for example 4:
{output_2}

## Example 5

### Instruction 5:
{instruction}

### Input 5:
{input}

### Output (a) for example 5:
{output_1}

### Output (b) for example 5:
{output_2}

## Example 6

### Instruction 6:
{instruction}

### Input 6:
{input}

### Output (a) for example 6:
{output_1}

### Output (b) for example 6:
{output_2}

## Example 7

### Instruction 7:
{instruction}

### Input 7:
{input}

### Output (a) for example 7:
{output_1}

### Output (b) for example 7:
{output_2}

## Preferred Output for example 3-7:
Now give all the preferred outputs for the batch of 5 examples.
<|im_end|>"""