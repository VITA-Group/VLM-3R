import re
import json
from tqdm import tqdm
import mmengine
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE")
]


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def classify_question_type(question):
    question_lower = question.lower()
    
    # Color questions
    if question_lower.startswith('what color') or question_lower.startswith('what is the color'):
        return 'Color'
    
    # Object nature questions  
    if (question_lower.startswith('what type') or 
        question_lower.startswith('what shape') or 
        question_lower.startswith('what kind')):
        return 'Object nature'
    
    # Place questions
    if question_lower.startswith('where is'):
        return 'Place'
    
    # Number questions
    if question_lower.startswith('how many'):
        return 'Number'
    
    # Object questions (What is, except color questions)
    if question_lower.startswith('what is') and not question_lower.startswith('what is the color'):
        return 'Object'
    
    # Everything else is Other
    return 'Other'

def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0

def calc_scanqa_score(preds, gts, tokenizer, scorers):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    
    # Initialize subtask tracking
    subtasks = ['Object', 'Color', 'Object nature', 'Place', 'Number', 'Other']
    subtask_stats = {task: {
        'count': 0,
        'acc': 0,
        'refined_acc': 0,
        'tmp_preds': {},
        'tmp_targets': {}
    } for task in subtasks}
    
    print("Total samples:", len(preds))
    
    # Create a dictionary to map question_id to ground truth data
    gt_dict = {gt['question_id']: gt for gt in gts}
    
    # Check if all predictions have corresponding ground truth
    missing_gt = [pred['question_id'] for pred in preds if pred['question_id'] not in gt_dict]
    if missing_gt:
        print(f"Warning: {len(missing_gt)} predictions have no corresponding ground truth")
        print(f"Missing question_ids: {missing_gt[:10]}...")  # Show first 10
    
    matched_count = 0
    for item_id, pred in tqdm(enumerate(preds)):
        question_id = pred['question_id']
        
        # Skip if no ground truth found
        if question_id not in gt_dict:
            continue
            
        gt = gt_dict[question_id]
        matched_count += 1
        
        # Classify question type
        question_type = classify_question_type(gt['question'])
        
        pred_answer = pred['text']
        gt_answers = gt['answers']
        pred_answer = clean_answer(pred_answer)
        ref_captions = [clean_answer(gt_answer) for gt_answer in gt_answers]
        tmp_acc, tmp_refined_acc = answer_match(pred_answer, ref_captions)
        
        # Update overall metrics
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred_answer}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
        
        # Update subtask metrics
        subtask_stats[question_type]['count'] += 1
        subtask_stats[question_type]['acc'] += tmp_acc
        subtask_stats[question_type]['refined_acc'] += tmp_refined_acc
        subtask_stats[question_type]['tmp_preds'][item_id] = [{'caption': pred_answer}]
        subtask_stats[question_type]['tmp_targets'][item_id] = [{'caption': caption} for caption in ref_captions]
    
    print(f"Matched {matched_count} predictions with ground truth")
    
    if matched_count == 0:
        print("Error: No matched predictions found!")
        return val_scores, subtask_stats
    
    # Calculate overall scores
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    acc = acc / matched_count
    refined_acc = refined_acc / matched_count
    val_scores["[scanqa] EM1"] = acc
    val_scores["[scanqa] EM1_refined"] = refined_acc
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scanqa] {m}"] = sc
        else:
            val_scores[f"[scanqa] {method}"] = score
    
    # Calculate subtask scores
    for task in subtasks:
        if subtask_stats[task]['count'] > 0:
            # Calculate EM scores
            subtask_stats[task]['em1'] = subtask_stats[task]['acc'] / subtask_stats[task]['count']
            subtask_stats[task]['em1_refined'] = subtask_stats[task]['refined_acc'] / subtask_stats[task]['count']
            
            # Calculate other metrics if there are samples
            if subtask_stats[task]['tmp_preds']:
                task_preds = tokenizer.tokenize(subtask_stats[task]['tmp_preds'])
                task_targets = tokenizer.tokenize(subtask_stats[task]['tmp_targets'])
                
                for scorer, method in scorers:
                    score, scores = scorer.compute_score(task_targets, task_preds)
                    if type(method) == list:
                        for sc, scs, m in zip(score, scores, method):
                            subtask_stats[task][m.lower()] = sc
                    else:
                        subtask_stats[task][method.lower()] = score
    
    return val_scores, subtask_stats

def print_subtask_results_table(subtask_stats):
    print("\n" + "="*80)
    print("SUBTASK EVALUATION RESULTS")
    print("="*80)
    
    # Print header with only the 5 requested metrics
    header = f"{'Question type':<15} {'Count':<8} {'BLEU-1':<8} {'BLEU-4':<8} {'METEOR':<8} {'ROUGE-L':<8} {'CIDEr':<8}"
    print(header)
    print("-" * 80)
    
    # Print results for each subtask
    for task in ['Object', 'Color', 'Object nature', 'Place', 'Number', 'Other']:
        if subtask_stats[task]['count'] > 0:
            bleu_1 = subtask_stats[task].get('bleu_1', 0) * 100 if 'bleu_1' in subtask_stats[task] else 0
            bleu_4 = subtask_stats[task].get('bleu_4', 0) * 100 if 'bleu_4' in subtask_stats[task] else 0
            meteor = subtask_stats[task].get('meteor', 0) * 100 if 'meteor' in subtask_stats[task] else 0
            rouge_l = subtask_stats[task].get('rouge_l', 0) * 100 if 'rouge_l' in subtask_stats[task] else 0
            cider = subtask_stats[task].get('cider', 0) * 100 if 'cider' in subtask_stats[task] else 0
            
            row = f"{task:<15} {subtask_stats[task]['count']:<8} {bleu_1:<8.2f} {bleu_4:<8.2f} {meteor:<8.2f} {rouge_l:<8.2f} {cider:<8.2f}"
            print(row)
        else:
            row = f"{task:<15} {subtask_stats[task]['count']:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}"
            print(row)
    
    print("-" * 80)
    
    # Calculate and print totals
    total_count = sum(subtask_stats[task]['count'] for task in subtask_stats)
    if total_count > 0:
        # Calculate weighted averages for the 5 requested metrics
        total_bleu_1 = sum(subtask_stats[task].get('bleu_1', 0) * subtask_stats[task]['count'] 
                          for task in subtask_stats if subtask_stats[task]['count'] > 0) / total_count * 100
        total_bleu_4 = sum(subtask_stats[task].get('bleu_4', 0) * subtask_stats[task]['count'] 
                          for task in subtask_stats if subtask_stats[task]['count'] > 0) / total_count * 100
        total_meteor = sum(subtask_stats[task].get('meteor', 0) * subtask_stats[task]['count'] 
                          for task in subtask_stats if subtask_stats[task]['count'] > 0) / total_count * 100
        total_rouge_l = sum(subtask_stats[task].get('rouge_l', 0) * subtask_stats[task]['count'] 
                           for task in subtask_stats if subtask_stats[task]['count'] > 0) / total_count * 100
        total_cider = sum(subtask_stats[task].get('cider', 0) * subtask_stats[task]['count'] 
                         for task in subtask_stats if subtask_stats[task]['count'] > 0) / total_count * 100
        
        total_row = f"{'TOTAL':<15} {total_count:<8} {total_bleu_1:<8.2f} {total_bleu_4:<8.2f} {total_meteor:<8.2f} {total_rouge_l:<8.2f} {total_cider:<8.2f}"
        print(total_row)
    
    print("="*80)

pred_json = 'results/llava_next_video_val_answer_pred_vsibench.json'
preds = [json.loads(q) for q in open(pred_json, "r")]
gt_json = 'thinking-in-space/data/scanqa/ScanQA_v1.0_val.json'
gts = mmengine.load(gt_json)

val_scores, subtask_stats = calc_scanqa_score(preds, gts, tokenizer, scorers)
print("\nOverall Scores:")
print(val_scores)

print_subtask_results_table(subtask_stats)