import pandas as pd
import numpy as np
import os

def generate_department_data(num_departments: int = 5, 
                           avg_department_size: int = 20,
                           size_variation: float = 0.3,
                           department_names: list = None,
                           output_file: str = "networks/data/DepartmentData.csv"):
    """
    Generate synthetic department data for UniversityNetwork.py
    
    Args:
        num_departments: Number of departments to generate
        avg_department_size: Average number of people per department
        size_variation: Coefficient of variation for department sizes (0.3 = 30% variation)
        department_names: List of department names (if None, will generate generic names)
        output_file: Path to save the generated CSV file
    
    Returns:
        pd.DataFrame: Generated department data
    """
    
    # Generate department names if not provided
    if department_names is None:
        department_names = list(range(1, num_departments + 1))
    
    # Generate department sizes using normal distribution
    std_dev = avg_department_size * size_variation
    department_sizes = np.random.normal(avg_department_size, std_dev, num_departments)
    
    # Ensure minimum size of 1 and convert to integers
    department_sizes = np.maximum(department_sizes, 1).astype(int)
    
    # Create the dataframe
    department_data = pd.DataFrame({
        'Department': department_names,
        'Size': department_sizes
    })
    
    # Create directory if it doesn't exist and save to CSV
    #os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #department_data.to_csv(output_file, index=False)
        
    return department_data

if __name__ == "__main__":
    generate_department_data(
         num_departments=8,
         avg_department_size=25,
         size_variation=0.4,
         department_names=["Dept A", "Dept B", "Dept C", "Dept D", "Dept E", "Dept F", "Dept G", "Dept H"]
     ) 