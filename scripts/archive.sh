torch-model-archiver --model-name total_model \
--version 1.0 \
--serialized-file '' \
--extra-files ./MyHandler.py,./model,./model_pth,./aiffel-gn3-2-035da204163f.json \
--handler my_handler.py  \
--export-path model-store -f \

