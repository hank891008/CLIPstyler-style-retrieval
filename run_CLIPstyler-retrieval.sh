python train_CLIPstyler-retrieval.py --content_path ./test_set/face2.jpeg --content_name face --exp_name exp1 --text "Oil painting of flowers"
python train_CLIPstyler-retrieval.py --content_path ./test_set/boat.jpg --content_name boat --exp_name exp2 --text "Oil painting of flowers"
python train_CLIPstyler-retrieval.py --content_path ./test_set/tubingen.jpeg --content_name house --exp_name exp1 --text "Oil painting of flowers"

python train_CLIPstyler-retrieval.py --content_path ./test_set/face2.jpeg --content_name face --exp_name exp2 --text "Starry  Night by Vincent van Gogh"
python train_CLIPstyler-retrieval.py --content_path ./test_set/boat.jpg --content_name boat --exp_name exp1 --text "Starry  Night by Vincent van Gogh" --device cuda:1
python train_CLIPstyler-retrieval.py --content_path ./test_set/tubingen.jpeg --content_name house --exp_name exp1 --text "Starry  Night by Vincent van Gogh"""

python train_CLIPstyler-retrieval.py --content_path ./test_set/face2.jpeg --content_name face --exp_name exp1 --text "The great wave off kanagawa by Hokusai"
python train_CLIPstyler-retrieval.py --content_path ./test_set/boat.jpg --content_name boat --exp_name exp1 --text "The great wave off kanagawa by Hokusai"
python train_CLIPstyler-retrieval.py --content_path ./test_set/tubingen.jpeg --content_name house --exp_name exp1 --text "The great wave off kanagawa by Hokusai"