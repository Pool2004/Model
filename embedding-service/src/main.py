import json
import logging
from confluent_kafka import Consumer, Producer, KafkaError
from src.config import settings
from src.embedder import embedder_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    consumer_config = {
        'bootstrap.servers': settings.KAFKA_BROKER,
        'group.id': settings.KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest'
    }
    producer_config = {
        'bootstrap.servers': settings.KAFKA_BROKER
    }
    
    consumer = Consumer(consumer_config)
    producer = Producer(producer_config)
    
    consumer.subscribe([settings.KAFKA_TOPIC_RAW])
    logger.info(f"Embedding Service consuming from {settings.KAFKA_TOPIC_RAW}")

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
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                
                if not text:
                    logger.warning(f"Empty text for doc_id {doc_id}")
                    continue
                
                # Generar vector
                logger.info(f"Generating embedding for doc_id: {doc_id}")
                vector = embedder_instance.embed_text(text)
                
                # Preparar mensaje emitido
                payload = {
                    "doc_id": doc_id,
                    "text": text,
                    "vector": vector,
                    "metadata": metadata
                }
                
                # Producir al tópico de embeddings
                producer.produce(
                    settings.KAFKA_TOPIC_EMBEDDINGS,
                    key=doc_id.encode('utf-8'),
                    value=json.dumps(payload).encode('utf-8'),
                    callback=delivery_report
                )
                producer.poll(0)
                
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from message")
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    main()
