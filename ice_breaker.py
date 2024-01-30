import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from output_parsers import person_intel_parser, PersonIntel
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from typing import Tuple


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)

    summary_template = """
             given the Linkedin information {information} about a person from I want you to create:
             1. a short summary
             2. two interesting facts about them
             3. topics of interest 
             4. ice breakers to talk to them 
             \n{format_instructions}
         """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={"format_instructions": person_intel_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    result = chain.run(information=linkedin_data)

    print(person_intel_parser.parse(result))

    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == '__main__':
    load_dotenv()
    print(os.environ['OPENAI_API_KEY'])
    result = ice_break('Aalen Talukdar TCS')

    # linkedin_profile_url = linkedin_lookup_agent(name=name)
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    #
    # twitter_username = twitter_lookup_agent(name=name)
    # tweets = scrape_user_tweets(username=twitter_username, num_tweets=5)
    #
    # summary_prompt = """
    # given the LinkedIn {information} about a person I want you to create :
    # 1. A short summary
    # 2. Two Interesting fact about them
    # 3. A topic that may interest them
    # 4. 2 creative Ice breakers to open a conversation with them
    # """
    #
    # summary_prompt_template = PromptTemplate(input_variables=["linkedin_information", "twitter_information"],
    #                                          template=summary_prompt)
    #
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #
    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    #
    # # result = chain.invoke(input = {"information" : information})
    #
    # print(chain.run(linkedin_information=linkedin_data, twitter_information=tweets))
