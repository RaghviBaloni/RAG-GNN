import re
from sklearn.metrics import f1_score, confusion_matrix

# Function to transform and normalize answer strings
def normalize_answer(input_string):
    return input_string.strip().lower().replace('|', '')

# Function to extract and normalize prediction labels
def extract_prediction(prediction_string):
    match_prediction = re.search(r"<prediction>(.*?)</prediction>", prediction_string, re.DOTALL)
    if match_prediction:
        return [label.strip().lower() for label in match_prediction.group(1).split()]
        #return match_prediction.group(1).strip().lower()
    return []

# Function to read predictions and answers from the file
def read_predictions(file_path, num_lines=None):
    predictions = []
    answers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Skip the header line
        lines = lines[1:]
        
        buffer = []
        count = 0
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                buffer.append(line)
            if '\t' in line:  # Found the answer line
                pred_lines = buffer[:-1]
                answer_line = buffer[-1]
                
                pred_line = " ".join(pred_lines)
                if '\t' not in answer_line:
                    print(f"Skipping line without tab character: {answer_line}")
                    buffer = []
                    continue
                
                pred, ans = answer_line.split('\t', 1)  # Use 1 to ensure we don't split more than once
                prediction = extract_prediction(pred_line)
                answer = normalize_answer(ans)
                
                predictions.append(prediction)
                answers.append(answer)
                
                buffer = []  # Reset buffer for next prediction-answer pair

                count += 1
                if num_lines and count >= num_lines:
                    break
    
    return predictions, answers

# Function to calculate and print evaluation metrics
def evaluate(predictions, answers, labels_of_interest):

    # Transform predictions to be single-label if they contain the true label
    transformed_predictions = []
    for pred, ans in zip(predictions, answers):
        if ans in pred:
            transformed_predictions.append(ans)
        else:
            transformed_predictions.append(pred[0] if pred else '')
    
    # Calculate F1 scores
    f1_micro = f1_score(answers, transformed_predictions, average='micro')
    f1_macro = f1_score(answers, transformed_predictions, average='weighted')

    # Print F1 scores
    print("Micro F1 Score:", f1_micro)
    print("Macro F1 Score:", f1_macro)

    # Print confusion matrix 
    print("Confusion Matrix:")
    print(confusion_matrix(answers, transformed_predictions, labels=labels_of_interest, normalize='true'))

# Main function to execute the evaluation
def main():
    file_path = 'dataset/goodreads/result/prediction.txt'  # Path to your prediction file
    labels_of_interest = ['non-fiction', 'fiction', 'romance']  # Define your labels of interest

    # Read the entire file for evaluation
    predictions, answers = read_predictions(file_path, num_lines = 2000)
    
    # Print the first 10 predictions and answers
    print("First 10 Predictions and Answers:")
    for i in range(len(predictions)):
        print(f"Prediction: {predictions[i]}, Answer: {answers[i]}")    

    # Evaluate the predictions
    evaluate(predictions, answers, labels_of_interest)

if __name__ == "__main__":
    main()

