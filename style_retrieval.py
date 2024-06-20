
import lancedb
from PIL import Image
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector

registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()

class Item(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

    @property
    def image(self):
        return Image.open(self.image_uri)

def init_dataset():
    db = lancedb.connect(".lancedb")
    if "item" in db:
        table = db["item"]
    else:
        import pandas as pd
        from pathlib import Path
        table = db.create_table("item", schema=Item)
        p = Path("archive").expanduser()
        uris = [str(f) for f in p.glob("*/*.jpg")]
        table.add(pd.DataFrame({"image_uri": uris}))
        table = db["item"]
    return table

def search(table, prompt, limit=3):
    import matplotlib.pyplot as plt
    rs = table.search(prompt).metric("cosine").limit(limit).to_list()
    best_score = 1 - rs[0]['_distance']
    selected = [rs[0]]
    for i in range(1, len(rs)):
        if 1 - rs[i]['_distance'] > best_score * 0.99:
            selected.append(rs[i])
        else:
            break
    for i in selected:
        plt.subplot(1, len(selected), selected.index(i)+1)
        plt.imshow(Image.open(i['image_uri']))
    plt.savefig(f"output_reference/{prompt.replace(' ', '_')}.jpg")
    return selected

def visualize(table, prompt):
    import matplotlib.pyplot as plt
    rs = table.search(prompt).metric("cosine").limit(1).to_list()
    img = Image.open(rs[0]['image_uri'])
    img = img.resize((512, 512))
    img.save("demo.jpg")

if __name__ == "__main__":
    table = init_dataset()
    prompt = input("Enter a prompt: ")
    visualize(table, prompt)