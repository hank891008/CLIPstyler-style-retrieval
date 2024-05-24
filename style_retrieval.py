
import lancedb
from PIL import Image
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector

registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()

class Pets(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

    @property
    def image(self):
        return Image.open(self.image_uri)
    
db = lancedb.connect(".lancedb")

import pandas as pd
from pathlib import Path
if "pets" in db:
    table = db["pets"]
else:
    table = db.create_table("pets", schema=Pets)
    p = Path("archive").expanduser()
    uris = [str(f) for f in p.glob("*/*.jpg")]
    table.add(pd.DataFrame({"image_uri": uris}))
    

# prompt = input("Enter a prompt: ")
# rs = table.search(prompt).metric("cosine").limit(3).to_pydantic(Pets)
# print(rs)
# r = table.search(prompt).metric("cosine").limit(3).to_list()
# for i in r:
#     print(i['image_uri'], 1 - i['_distance'])
# import matplotlib.pyplot as plt
# for i in range(len(rs)):
#     plt.subplot(1, len(rs), i+1)
#     plt.imshow(rs[i].image)
# plt.show()

def init_dataset():
    db = lancedb.connect(".lancedb")
    table = db["pets"]
    return table

def search(table, prompt):
    import matplotlib.pyplot as plt
    rs = table.search(prompt).metric("cosine").limit(3).to_pydantic(Pets)
    plt.imshow(rs[0].image)
    plt.savefig(f"outputs/{prompt.replace(' ', '_')}.jpg")
    rs = table.search(prompt).metric("cosine").limit(3).to_list()
    
    return rs


if __name__ == "__main__":
    table = init_dataset()
    # prompt = input("Enter a prompt: ")
    # rs = search(table, prompt)
    # import numpy as np
    # print(np.array(rs[0]["vector"]).shape)
    prompt = input("Enter a prompt: ")
    rs = table.search(prompt).metric("cosine").limit(3).to_pydantic(Pets)
    print(rs)
    r = table.search(prompt).metric("cosine").limit(3).to_list()
    for i in r:
        print(i['image_uri'], 1 - i['_distance'])
    import matplotlib.pyplot as plt
    for i in range(len(rs)):
        plt.subplot(1, len(rs), i+1)
        plt.imshow(rs[i].image)
    plt.show()