#!/usr/bin/env python3
"""
Data Analysis Script for Gemma Fine-Tuning Dataset

This script analyzes the JSONL files to help understand the data before training.
"""

import json
import os
from collections import Counter

DATA_DIR = "data"
DATA_FILES = [
    "welcoming_gemma_clean.jsonl",
    "query_question_gemma_clean.jsonl",
    "rule_lookup_gemma_clean.jsonl",
    "out_of_scope_gemma_clean.jsonl",
]

def analyze_file(filepath):
    """Analyze a single JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return data

def main():
    print("=" * 70)
    print("Gemma Fine-Tuning Dataset Analysis")
    print("=" * 70)
    
    total_examples = 0
    type_counts = Counter()
    avg_question_length = 0
    avg_answer_length = 0
    
    for filename in DATA_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n{'=' * 70}")
        print(f"Analyzing: {filename}")
        print("=" * 70)
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        
        data = analyze_file(filepath)
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        print(f"📊 File Size: {file_size:.2f} KB")
        print(f"📝 Total Examples: {len(data)}")
        
        total_examples += len(data)
        
        # Analyze structure
        if data:
            sample = data[0]
            print(f"\n🔍 Sample Structure:")
            print(f"   Keys: {list(sample.keys())}")
            
            # Check output structure
            if 'output' in sample:
                print(f"   Output type: {type(sample['output'])}")
                if isinstance(sample['output'], dict):
                    print(f"   Output keys: {list(sample['output'].keys())}")
            
            # Count types
            file_types = []
            question_lengths = []
            answer_lengths = []
            
            for item in data:
                output = item.get('output', {})
                if isinstance(output, dict):
                    data_type = output.get('type', 'unknown')
                    file_types.append(data_type)
                    
                    # Calculate lengths
                    question = item.get('question', '')
                    question_lengths.append(len(question))
                    
                    answer = output.get('answer', '')
                    if answer:
                        answer_lengths.append(len(answer))
                    else:
                        # For non-answer types, use string representation
                        answer_lengths.append(len(str(output)))
                
            type_counts.update(file_types)
            
            if question_lengths:
                avg_q = sum(question_lengths) / len(question_lengths)
                avg_question_length += avg_q
                print(f"\n📏 Question Length:")
                print(f"   Average: {avg_q:.1f} characters")
                print(f"   Min: {min(question_lengths)} chars")
                print(f"   Max: {max(question_lengths)} chars")
            
            if answer_lengths:
                avg_a = sum(answer_lengths) / len(answer_lengths)
                avg_answer_length += avg_a
                print(f"\n💬 Answer/Output Length:")
                print(f"   Average: {avg_a:.1f} characters")
                print(f"   Min: {min(answer_lengths)} chars")
                print(f"   Max: {max(answer_lengths)} chars")
            
            # Show sample
            print(f"\n👀 Sample Example:")
            sample = data[0]
            print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
            if isinstance(sample.get('output'), dict):
                print(f"   Type: {sample['output'].get('type', 'N/A')}")
                answer = sample['output'].get('answer', str(sample['output']))
                print(f"   Answer: {answer[:100]}...")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"📊 Total Examples Across All Files: {total_examples}")
    print(f"\n📝 Type Distribution:")
    for data_type, count in type_counts.most_common():
        print(f"   {data_type}: {count} ({count/total_examples*100:.1f}%)")
    
    num_files = len([f for f in DATA_FILES if os.path.exists(os.path.join(DATA_DIR, f))])
    if num_files > 0:
        avg_q = avg_question_length / num_files
        avg_a = avg_answer_length / num_files
        print(f"\n📏 Average Question Length: {avg_q:.1f} characters")
        print(f"💬 Average Answer Length: {avg_a:.1f} characters")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
