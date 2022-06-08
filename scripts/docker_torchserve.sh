sudo docker run --rm -it \
-p 3000:9000 -p 3001:9001 -p 3002:9002 -p 3003:3903 \
-v $(pwd)/model-store:/home/model-server/model-store anfqlc0311/torchserve-mtail \
torchserve --start --ts-config model-store/config.properties --ncs --model-store model-store --models model=total_model.mar 
