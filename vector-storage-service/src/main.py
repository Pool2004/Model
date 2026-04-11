import json
import logging
from confluent_kafka import Consumer, KafkaError
from src.config import settings
from src.chroma_client import chroma_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    consumer_config = {
        'bootstrap.servers': settings.KAFKA_BROKER,
        'group.id': settings.KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest'
    }
    
    consumer = Consumer(consumer_config)
    consumer.subscribe([settings.KAFKA_TOPIC_EMBEDDINGS])
    logger.info(f"Vector Storage Service consuming from {settings.KAFKA_TOPIC_EMBEDDINGS}")

    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info('Reached end of partition')
                else:
                    logger.error(f"Error while consuming: {msg.error()}")
                continue

            try:
                # Procesar mensaje recibido
                val_str = msg.value().decode('utf-8')
                data = json.loads(val_str)
                
                doc_id = data.get("doc_id")
                text = data.get("text")
                vector = data.get("vector")
                metadata = data.get("metadata", {})
                
                if not doc_id or not vector or not text:
                    logger.warning(f"Incomplete payload for document. Missing id, vector or text.")
                    continue
                
                logger.info(f"Saving vector for doc_id: {doc_id}")
                success = chroma_instance.store_document(
                    doc_id=doc_id,
                    text=text,
                    vector=vector,
                    metadata=metadata
                )
                
                if not success:
                    logger.error(f"Failed to store doc_id {doc_id}")
                
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from message")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
