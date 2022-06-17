torch-model-archiver --model-name total_model \
--version 1.0 \
--serialized-file '' \
--extra-files ./MyHandler.py,./model,./model_pth,./key.json \
--handler my_handler.py  \
--export-path model-store -f \

