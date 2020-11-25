gcloud ai-platform jobs submit training MNIST_CV_20200413_0 \
--module-name=trainer.run_cv \
--runtime-version 1.15 \
--python-version 3.7 \
--package-path=./trainer \
--job-dir=gs://data_shikhar_skd \
--region=us-central1 \
--config=trainer/cloudml-gpu.yaml
