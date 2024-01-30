from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    template = """given the full name {name_of_person}, I want you to get me a link to their LinkedIn profile,
                        Your answer should only have a URL."""
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    tools_for_agent = [Tool(name='Crawl google for linkedin page',
                            func=get_profile_url,
                            description='useful for when you need to get the LinkedIn page url')]

    agent = initialize_agent(tools=tools_for_agent,
                             llm=llm,
                             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    prompt_template = PromptTemplate(template=template, input_variables=['name_of_person'])

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
