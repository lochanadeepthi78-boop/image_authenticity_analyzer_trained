IMAGE AUTHENTICITY ANALYZER - DATASET UPDATE

Important:
The uploaded build_dataset.py is a dataset-generation script, not an image dataset.
It expects external image pools and creates:
  dataset/real/       500 real images
  dataset/ai/         500 AI-generated images
  dataset/edited/     500 edited images
  dataset/labels.csv

The current project dataset remains in the dataset/ folder because no replacement
image dataset was uploaded. The web backend currently uses handcrafted image
analysis in analyzer.py; it does not train on or load these dataset images.

To generate a replacement dataset, place your source images in:
  real_photos/
  ai_images/

Then run:
  python build_dataset.py --real_pool ./real_photos --ai_pool ./ai_images --out ./dataset

Do not delete analyzer.py or app.py. The frontend is in templates/ and static/.
