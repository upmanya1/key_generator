"""
Keyword Generator for SEO Campaigns
Uses Google Gemini AI to generate targeted keywords from website crawl data
"""

import argparse
import csv
import re
import dotenv
import time
from urllib.parse import urlparse
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

class WebsiteAssistant:
    """AI-powered assistant for analyzing website content and generating marketing insights"""
    
    def __init__(self, llm):
        self.context = ""  # Combined website content
        self.context_dict = {}  # URL -> page content mapping
        self.llm = llm  # Google Gemini LLM instance
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = None  # LangChain conversation chain
        self.qa_responses = []  # Storage for Q&A history

    def _clean_text(self, text):
        """Sanitize text content by removing excessive whitespace"""
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def initialize_chat_chain(self):
        """Set up the conversational AI chain with system prompt"""
        SYSTEM_PROMPT = (
            f"You are a website expert assistant. Use this full context:\n{self.context}\n"
            "Answer strictly based on this context, cite URLs, and be concise."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

    def chat(self, question):
        """Execute a question against the AI chain"""
        try:
            response = self.chain.invoke({"question": question})
            return response['text']
        except Exception as e:
            return f"Error: {e}"

def load_crawler_output(path: str) -> pd.DataFrame:
    """Load and validate crawler CSV output"""
    df = pd.read_csv(path)
    if 'URL' not in df.columns or 'Page Text' not in df.columns:
        raise ValueError("Input CSV must contain 'URL' and 'Page Text' columns.")
    return df

def generate_keywords(llm, context: str, web_context: dict):
    """Generate SEO keywords using multi-step AI analysis"""
    assistant = WebsiteAssistant(llm)
    assistant.context = context
    assistant.context_dict = web_context
    assistant.initialize_chat_chain()

    # Strategic questions to extract business details
    questions = [
        "What is the name of the main company described on the website and its business type?",
        "List every single product or services with all names offered by the company with brief details which is required for a google ad manager."
    ]
    
    # Get contextual answers from AI
    answers = {}
    for q in questions:
        answers[q] = assistant.chat(q)
        time.sleep(1)  # Rate limit protection
    
    # Build context for keyword generation
    answers_context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items()])
    
    # Custom SEO prompt template
    custom_prompt = f"""
    You are an expert SEO analyst. Use the following answered context:
    {answers_context}

    Generate high-ROI, long-tail Google Ads keywords focused on these products which user may think. 
    Include action words (buy, order, best, discount), and append each keyword with its source URL 
    (be careful with URL, URL should be intact) in parentheses.
    Return a comma-separated list only.
    """
    
    # Execute final keyword generation
    prompt = ChatPromptTemplate.from_messages([
        ("system", custom_prompt),
        ("human", "Generate keywords:")
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({})
    return [k.strip() for k in response['text'].split(',') if k.strip()]

def main():
    """Main execution flow"""
    parser = argparse.ArgumentParser(description="Generate SEO keywords from crawler CSV output.")
    # Input/output configurations
    parser.add_argument("--input", "-i", default='results.csv', help="Path to crawler CSV file.")
    parser.add_argument("--output", "-o", default="keywords_output.csv", help="Output CSV path.")
    # Model configurations
    parser.add_argument("--model", "-m", default="gemini-2.0-flash-thinking-exp-01-21",
                       help="Google GenAI model name.")
    parser.add_argument("--api-key", "-k", help="Your Google GenAI API key.")
    parser.add_argument("--delay", "-d", type=float, default=1.0,
                       help="Delay between LLM calls to avoid rate limits.")
    args = parser.parse_args()

    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=1,  # Higher temperature for creative output
        api_key=args.api_key,
    )

    # Load and validate crawled data
    df = load_crawler_output(args.input)
    web_context = {row['URL']: row['Page Text'] for _, row in df.iterrows()}
    all_text = "\n\n".join([f"URL: {url}\n{text}" for url, text in web_context.items()])

    # Generate keywords using AI analysis
    custom_keywords = generate_keywords(llm, all_text, web_context)

    # Save results with keyword-URL mapping
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['keyword', 'URL'])
        for item in custom_keywords:
            if '(' in item and item.endswith(')'):
                kw_text = item[:item.rfind('(')].strip()
                url = item[item.rfind('(')+1:-1].strip()
            else:
                kw_text = item
                url = ''
            writer.writerow([kw_text, url])

    print(f"[✓] Generated {len(custom_keywords)} keywords. Output saved to {args.output}")

if __name__ == "__main__":
    main()
