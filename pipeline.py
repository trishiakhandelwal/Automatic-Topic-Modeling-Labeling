import pandas as pd
import numpy as np
from openai import OpenAI
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

SUMMARIES_FILE = "summaries.csv"
TOKEN_LIMIT = 6000  # safety cap per doc

#User must enter their own openai api key
client = OpenAI(api_key="[ENTER KEY HERE]")

prompt = """
You are a news summarization expert. Write a brief summary (60–80 words) that captures the full context and main theme of the document.

input: "Musicians groups are to tackle US visa regulations which are blamed for hindering British acts chances of succeeding across the Atlantic. A singer hoping to perform in the US can expect to pay $1,300 (£680) simply for obtaining a visa. Groups including the Musicians Union are calling for an end to the 'raw deal' faced by British performers. US acts are not faced with comparable expense and bureaucracy when visiting the UK for promotional purposes. Nigel McCune from the Musicians Union said British musicians are 'disadvantaged' compared to their US counterparts. A sponsor has to make a petition on their behalf, which is a form amounting to nearly 30 pages, while musicians face tougher regulations than athletes and journalists. 'If you make a mistake on your form, you risk a five-year ban and thus the ability to further your career,' says Mr McCune. 'The US is the worlds biggest music market, which means something has to be done about the creaky bureaucracy,' says Mr McCune. 'The current situation is preventing British acts from maintaining momentum and developing in the US,' he added. The Musicians Union stance is being endorsed by the Music Managers Forum (MMF), who say British artists face 'an uphill struggle' to succeed in the US, thanks to the tough visa requirements, which are also seen as impractical. The MMFs general secretary James Seller said: 'Imagine if you were an orchestra from the Orkneys? Every member would have to travel to London to have their visas processed.' 'The US market is seen as the holy grail and one of the benchmarks of success, and were still going to fight to get in there. 'Its still very important, but there are other markets like Europe, India and China,' added Mr Seller. A Department for Media, Culture and Sport spokeswoman said: 'Were aware that people are experiencing problems, and are working with the US embassy and record industry to see what we can do about it.' A US Embassy spokesman said: 'We are aware that entertainers require visas for time-specific visas and are doing everything we can to process those applications speedily.' 'We are aware of the importance of cultural exchange and we will do our best to facilitate that.'",
output: ("British musicians face significant challenges due to costly and complex US visa regulations, unlike their US counterparts who have easier access to the UK. The Musicians Union and Music Managers Forum are advocating for reform, arguing that these barriers hinder British acts from succeeding in the US market. Both the UK government and the US embassy acknowledge the issue, with efforts underway to address the bureaucratic hurdles and facilitate cultural exchange.")

input: "Fuming Robinson blasts officials  England coach Andy Robinson insisted he was "livid" after his side were denied two tries in Sundays 19-13 Six Nations loss to Ireland in Dublin.  Mark Cuetos first-half effort was ruled out for offside before the referee spurned TV replays when England crashed over in the dying minutes. "[Im] absolutely spitting. Im livid. Theres two tries weve been cost," Robinson told BBC Sport. "Weve got to go back to technology. I dont know why we didnt." South African referee Jonathan Kaplan ruled that Cueto was ahead of Charlie Hodgson when the fly-half hoisted his cross-field kick for the Sale wing to gather.  Kaplan then declined the chance to consult the fourth official when Josh Lewsey took the ball over the Irish line under a pile of bodies for what could have been the game-winning try. "I think Mark Cueto scored a perfectly legal try and I think he should have gone to the video referee on Josh Lewsey," said Robinson. "It is how we use the technology. It is there, and it should be used. "I am still trying to work out the Cueto try. I have looked at both, and they both looked tries. "We are very disappointed, and this will hurt, there is no doubt about that. "We are upset now, but the referee is in charge and he has called it his way and we have got to be able to cope with that.  "We did everything we could have done to win the game. I am very proud of my players and, with a couple of decisions, this could have been a very famous victory. "I thought we dominated. Matt Stevens had an awesome game at tighthead prop, while the likes of Charlie Hodgson, Martin Corry and Lewis Moody all came through well. "Josh Lewsey was awesome, and every one of the forwards stood up out there. Given the pressure we were under, credit must go to all the players. "We have done everything but win a game of rugby, but Ireland are a good side. They defended magnificently and theyve got every chance of winning this Six Nations." England have lost their first three matches in this years Six Nations and four out of their six games since Robinson took over from Sir Clive Woodward in September.",
output: ("England coach Andy Robinson expressed outrage after his team was denied two tries in their 19-13 Six Nations defeat to Ireland. Robinson criticized referee Jonathan Kaplan for not using TV replays to verify the tries, which he believes were legitimate. Despite the loss, Robinson praised his players' efforts and resilience. England has now lost their first three matches in this year's Six Nations, continuing a challenging season since Robinson's tenure began.")

input: """

def truncate_to_n_tokens(text, n=TOKEN_LIMIT):
        if pd.isna(text):
            return ""
        text = str(text)  # ensure it’s a string
        tokens = text.split()
        return " ".join(tokens[:n])

#Generates summary for one document using a few shot prompt for the gpt-4o-mini model
def generate_summary(doc: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Input: {doc}\nOutput:"}
            ],
            max_tokens=300  
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing doc: {e}")
        return ""
    
# Finds the top 10 most representative documents for each topic to help with label generation
def find_top_docs(top_words, documents_per_topic):
    top_documents_per_topic = {}

    # print(top_words)
    for topic_id, words in top_words.items():
        topic_docs = documents_per_topic.get(topic_id, pd.DataFrame()).reset_index(drop=True)
        # print(topic_docs)
        vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english")
        representation_model = MaximalMarginalRelevance(diversity=0.1)

        indiv_model = BERTopic(
            vectorizer_model=vectorizer_model, 
            top_n_words=10, 
            min_topic_size=10, 
            representation_model=representation_model
        )
        indiv_topics, _ = indiv_model.fit_transform(topic_docs.content)

        indiv_doc_info = indiv_model.get_document_info(topic_docs.content)
        indiv_topics_info = indiv_model.get_topic_info()

        largest_subtopic_id = indiv_topics_info.loc[
            indiv_topics_info['Count'] == indiv_topics_info['Count'].max(), 'Topic'
        ].values[0]

        topic_docs_for_model = topic_docs.rename(columns={'content':'Document'})
        largest_subtopic_docs = indiv_doc_info[indiv_doc_info['Topic'] == largest_subtopic_id]
        largest_subtopic_docs['content'] = largest_subtopic_docs['Document'].map(
            dict(zip(topic_docs['content'], topic_docs['content']))
        )

        selected_docs = largest_subtopic_docs.sample(n=min(10, len(largest_subtopic_docs)), random_state=42)
        top_documents_per_topic[topic_id] = selected_docs[['content']]

    return top_documents_per_topic

# Uses the top 10 representative documents from each topic to generate a label using persona-based prompting
def name_generator(top_docs_by_topic):
    name_generation_prompt = """You are an expert at coming up with themes based on summaries of news articles. 
    You will be given 10 news summaries and your task is to generate a name for this group that would represent the common theme 
    of all the summaries. This name should be between 2-7 words."""

    df_list = []
    for topic_id, df in top_docs_by_topic.items():
        df['Topic'] = topic_id  # Add the topic ID as a column
        df_list.append(df)
    
    df_top_documents = pd.concat(df_list, ignore_index=True)
    df_top_documents['content'] = df_top_documents['content'].str.replace(r'[()]', '', regex=True)
    df_top_documents = df_top_documents[['content', 'Topic']]

    Topic = []
    Name = []
    for i in df_top_documents.Topic.unique():
        docs = df_top_documents[df_top_documents['Topic'] == i].content.tolist()
        docs_text = "\n".join([f"- {doc}" for doc in docs])
        messages = [
            {"role": "system", "content": name_generation_prompt},
            {"role": "user", "content": f"Here are the summaries:\n{docs_text}\nPlease provide a topic name:"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50
        )
        topic_name = response.choices[0].message.content.strip()
        Topic.append(i)
        Name.append(topic_name)

    df_name = pd.DataFrame({
        'Topic': Topic,
        'Name': Name
    })

    return df_name

# Runs the entire backend pipeline for use in app.py
def run_pipeline(input_file):
    """
    Runs the full summarization and topic modeling pipeline on the given CSV file.

    Returns:
        df (pd.DataFrame): Original data with summaries and assigned topics.
        topic_docs_mapping (dict): {topic_id: pd.DataFrame} with 'summary' and 'content'
        df_topic_names (pd.DataFrame): Topic names {Topic, Name}
    """
    df = pd.read_csv(input_file, on_bad_lines='skip')
    df = df[['content']].copy()
    df.columns = ["content"]

    # Truncate text
    df["truncated_text"] = df["content"].apply(truncate_to_n_tokens)

    # Generate summaries
    summaries = []
    for i, doc in enumerate(df["truncated_text"], 1):
        print(f"Summarizing doc {i}/{len(df)}")
        summary = generate_summary(doc)
        summaries.append(summary)
    df["summary"] = summaries

    df.to_csv(SUMMARIES_FILE, index=False)

    # BERTopic setup
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english")
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    representation_model = MaximalMarginalRelevance(diversity=0.1)

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        min_topic_size=10,
        representation_model=representation_model,
        embedding_model=embed_model
    )

    # Fit and transform summaries
    topics, _ = topic_model.fit_transform(df["summary"].tolist())
    df["topic"] = topics
    embeddings = np.array(topic_model._extract_embeddings(df["summary"].tolist()))

    # Assign all documents to their categories in a dict
    documents_per_topic = {}
    for topic_id in set(topics):
        topic_docs = df.loc[df["topic"] == topic_id, ["summary"]].copy()
        topic_docs.rename(columns={"summary": "content"}, inplace=True)
        documents_per_topic[int(topic_id)] = topic_docs

    # Get top words per topic
    top_words = {topic: [word for word, _ in topic_model.get_topic(topic)]
                 for topic in set(topics) if topic != -1}

    # Find top documents per topic (using summaries)
    top_docs_per_topic = find_top_docs(top_words, documents_per_topic)

    if -1 in documents_per_topic:
        top_docs_per_topic[-1] = documents_per_topic[-1]

    # Generate topic names, using "Miscellaneous" for -1
    df_topic_names = name_generator(top_docs_per_topic)
    if -1 in df_topic_names["Topic"].values:
        df_topic_names.loc[df_topic_names["Topic"] == -1, "Name"] = "Miscellaneous"

    return df, documents_per_topic, df_topic_names, embeddings

# Using text embeddings to find documents most similar to the query
def query_documents(query_text, df, embeddings, top_n=5):
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = embed_model.encode([query_text])[0]

    sims = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_idxs = np.argsort(sims)[::-1][:top_n]

    # Keep summary for cards and full document for popup
    results = df.iloc[top_idxs].copy()
    results["content"] = results["summary"]           # summary for card
    results["full_content"] = df.iloc[top_idxs]["content"].values  # original full document

    return results