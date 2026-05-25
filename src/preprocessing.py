import pandas as pd

def preprocess_input(data):

    # Convertir sexo
    data["Sex"] = data["Sex"].map({
        "female": 0,
        "male": 1
    })

    # Variables dummy para Embarked
    data["Embarked_Q"] = (
        data["Embarked"] == "Q"
    ).astype(int)

    data["Embarked_S"] = (
        data["Embarked"] == "S"
    ).astype(int)

    # Eliminar columna original
    data.drop("Embarked", axis=1, inplace=True)

    return data