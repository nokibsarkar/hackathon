import re
from pprint import pprint as print

def parse_recipe_file(file_path: str):
    recipes = []
    recipe_pattern = r"(?m)^\d+\.\sRecipe\s*-\s*(.+):\nprocess:\s*(.*?)\ningredients:\s*(.+)$"
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        matches = re.finditer(recipe_pattern, content)
        for match in matches:
            title = match.group(1).strip()
            process = match.group(2).strip()
            ingredients = [ingredient.strip() for ingredient in match.group(3).split(",")]
            
            recipes.append({
                "title": title,
                "process": process,
                "ingredients": list(sorted(ingredients))
            })
    return recipes

"""
I want something sweet today.

recipe1: ing: [1,2,3]
recipe2: ing: [2,4,50]


Query: 
db->ing: 

recipe1: ing: [1,2,3]
recipe2: ing: [2,4,5]


recipe3: ing: [2, 30]
"""
if __name__ == "__main__":
    recipes = parse_recipe_file("static/recipe.txt")
    print(recipes)