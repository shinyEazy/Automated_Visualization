import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import dotenv

dotenv.load_dotenv()

endpoint = "https://models.github.ai/inference"
model = "deepseek/DeepSeek-V3-0324"
token = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        UserMessage(
            """
task_description:
  type: Text classification
  description: |
    Given a text passage authored by a person, this task aims to identify the underlying emotion expressed in the text. 
    The model classifies the emotion into one of the following categories: anger, disgust, fear, joy, neutral, sadness, or surprise. 
    The classification is based solely on the semantic and emotional cues present in the input text.
  input: |
    A single text passage written by an individual. The text may express an emotional state either explicitly (e.g., "I am so happy today!") or implicitly (e.g., "The sun finally came out after days of rain."). 
    The passage should be in natural language and may vary in length.

  output: |
    A single emotion label representing the primary emotion conveyed by the author in the text. 
    The possible labels are one of the following: anger, disgust, fear, joy, neutral, sadness, or surprise.

  visualize:
    description: |
      Display a list of input data along with the predicted emotion results. Each data item includes:
        - The input text passage to be classified.
        - The predicted emotion based on the text.
        - The probabilities of all possible emotions.
        - An emoji corresponding to the predicted emotion.

    features:
      - list_display:
          description: Show a list of input data and their prediction results.
          fields:
            - input_text: The input text passage.
            - predicted_emotion: The predicted emotion.
            - emotion_probabilities: Probabilities of all possible emotions.
            - emotion_emoji: Emoji corresponding to the predicted emotion.
      - input_function:
          description: Allow users to enter new text for emotion prediction.
          steps:
            - Enter a text passage.
            - Display the prediction result (emotion, emotion probabilities, emoji).

model_information:
  api_url: "http://34.87.113.245:8000/api/text-classification"
  name: j-hartmann/emotion-english-distilroberta-base
  description: The model was trained on 6 diverse datasets and predicts Ekman's 6 basic emotions, plus a neutral class.
  input_format: 
    type: json
    structure:
      texts:
        type: string
        description: A text passage written by the author.
  output_format: 
    description: A list of dict contains emotions (labels) and their corresponding scores (probabilities).
    type: List[dict]
    structure:
      label: 
        type: string
      score: 
        type: float
  parameters:
    config:
      id2label:
        "0": anger
        "1": disgust
        "2": fear
        "3": joy
        "4": neutral
        "5": sadness
        "6": surprise
      label2id:
        anger: 0
        disgust: 1
        fear: 2
        joy: 3
        neutral: 4
        sadness: 5
        surprise: 6

dataset_description:
  description: GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 6 emotion categories or Neutral.
  data_source: ./data/goemotions.csv file
  data_format: 
    text: The text of the comment (with masked tokens, as described in the paper).
    id: The unique id of the comment.
    author: The Reddit username of the comment's author.
    subreddit: The subreddit that the comment belongs to.
    link_id: The link id of the comment.
    parent_id: The parent id of the comment.
    created_utc: The timestamp of the comment.
    rater_id: The unique id of the annotator.
    example_very_unclear: Whether the annotator marked the example as being very unclear or difficult to label (in this case they did not choose any emotion labels).
    emotions:
      anger: Binary label (0 or 1)
      disgusts: Binary label (0 or 1)
      fear: Binary label (0 or 1)
      joy: Binary label (0 or 1)
      neutral: Binary label (0 or 1)
      sadness: Binary label (0 or 1)
      surprise: Binary label (0 or 1)


Constraints:
      You are a helpful assistant that can answer questions and help with tasks.
      Your task is to generate a UI for the task.
      - The UI must be in HTML format and use Tailwind CSS.
      - The UI must be responsive, interactive, easy to use, and visually appealing.
      - The UI must use the real API endpoint for all emotion predictions. Do NOT use any mock data, static responses, or local simulation.
      - For every prediction, the UI must send a POST request to the following API endpoint:
        - API URL: http://34.87.113.245:8000/api/text-classification
        - Input format:
          {
            "texts": "The text passage to be classified."
          }
        - Output format:
          {
            "code": "000",
            "message": "Thành công",
            "data": [
              [
                {"label": "anger", "score": ...},
                {"label": "disgust", "score": ...},
                {"label": "fear", "score": ...},
                {"label": "joy", "score": ...},
                {"label": "neutral", "score": ...},
                {"label": "sadness", "score": ...},
                {"label": "surprise", "score": ...}
              ]
            ]
          }
      - The UI must display:
        - The input text passage.
        - The predicted emotion (the label with the highest score).
        - The probabilities (scores) for all possible emotions.
        - An emoji corresponding to the predicted emotion.
      - The UI must allow users to enter new text for emotion prediction and display the result using the real API.
      - Add code comments in the HTML/JS to indicate where the API call is made.

Important:
      - Do not use any mock API, mock data, or hardcoded results.
      - All prediction results must come from the real API endpoint above.
"""
        ),
    ],
    max_tokens=8000,
    model=model
)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)
print("Response saved to response.txt")