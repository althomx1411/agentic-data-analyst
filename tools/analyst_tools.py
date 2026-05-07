
import os

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import json 
import re
import sys 
import textwrap
from pathlib import Path
from IPython.display import display, Image as IPImage
import ipywidgets as widgets
import numpy as np
from collections import defaultdict

def load_data(file_path: str)-> dict:
   
    df = pd.read_csv(file_path, encoding='latin-1')

    columns = df.columns

    for col in columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    schema = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": {}
    }

    for col in columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "nulls": df[col].isnull().sum(),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = df[col].min().item() if not df[col].isna().all() else None
            col_info["max"] = df[col].max().item() if not df[col].isna().all() else None
            col_info["mean"] = df[col].mean().item() if not df[col].isna().all() else None

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["min"] = str(df[col].min()) if df[col].isna().all() else None
            col_info["max"] = str(df[col].max()) if df[col].isna().all() else None    
        
        else:
            top_values = df[col].value_counts().head(5)
            col_info["top_values"] = top_values.to_dict()
    

        schema["column_info"][col] = col_info



    return {"dataframe": df, "schema": schema}
                

def run_eda(df: dict) -> dict:

    answer = { 
        "shape" : df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}

    }


    numeric_cols = df.select_dtypes(include=['number']).columns
   
    for col in numeric_cols:
        answer["numeric_summary"][col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }

    categorica_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorica_cols:
        top_values = df[col].value_counts().head(5)
        answer["categorical_summary"][col] = {
            "top_values": top_values.to_dict(),
            "unique_values": df[col].nunique()
        }  
    return answer 

def generate_chart(df, type, x, y=None):
    plt.figure(figsize=(10, 6))
    
    if type == 'bar':
        if y:
            sns.barplot(data=df, x=x, y=y)
        else:
            sns.countplot(data=df, x=x)
    elif type == 'line':
        sns.lineplot(data=df, x=x, y=y)
    elif type == 'scatter':
        sns.scatterplot(data=df, x=x, y=y)
    elif type == 'hist':
        sns.histplot(data=df, x=x)
    else:
        raise ValueError("Unsupported chart type")
    
    # Labels and title
    plt.title(f"{type.capitalize()} Chart of {y} vs {x}" if y else f"{type.capitalize()} Chart of {x}")
    plt.xlabel(x)
    if y:
        plt.ylabel(y)
    
    plt.tight_layout()
    
    # Save
    chart_path = f"{type}_{x}_{y}.png" if y else f"{type}_{x}.png"
    plt.savefig(chart_path)
    plt.close()
    
    return {"chart_path": chart_path}


def run_python_snippet(code, df):
    try:
       
        namespace = {
            "df": df,
            "memory": _memory_store,  
            "update_memory": update_collection  
        }
        exec(code, namespace)
        
        
        result = namespace.get("_result", "Code executed successfully")
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
    


_memory_store = {}
_memory_file = "agent_memory.json"

def get_collection():
    return _memory_store.copy()

def update_collection(key, value):
    _memory_store[key] = value
    return {"success": True, "stored": key}

if os.path.exists(_memory_file):
    with open(_memory_file, 'r') as f:
        _memory_store = json.load(f)


