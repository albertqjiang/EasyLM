# This script converts the standrd LLaMA PyTorch checkpoint released by Meta
# to the EasyLM checkpoint format. The converted checkpoint can then be loaded
# by EasyLM for fine-tuning or inference.

# This script is largely borrow from https://github.com/Sea-Snell/JAX_llama

from pathlib import Path
import json
import numpy as np
import torch
import flax
import mlxu
import io

from EasyLM.checkpoint import StreamingCheckpointer

from smart_open import open as open
from google.cloud import storage

GCS_PATH_INDICATOR = "gs://"
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    checkpoint_dir='',
    output_file='',
    streaming=True,
)


def main(argv):
    checkpoint_dir = FLAGS.checkpoint_dir.rstrip("/").strip()
    if checkpoint_dir.startswith(GCS_PATH_INDICATOR):
        # On gcs
        parts = checkpoint_dir.lstrip(GCS_PATH_INDICATOR).split("/")
        gcs_bucket_name, gcs_prefix = parts[0], "/".join(parts[1:])

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(gcs_bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))

        ckpt_paths = []
        for blob in blobs:
            if blob.name.endswith(".pth"):
                ckpt_paths.append(
                    f"{GCS_PATH_INDICATOR}{blob.bucket.name}/{blob.name}"
                )
        ckpt_paths = sorted(ckpt_paths)
    else:
        ckpt_paths = sorted(Path(FLAGS.checkpoint_dir).glob("*.pth"))
    ckpts = {}
    for _, ckpt_path in enumerate(ckpt_paths):
        if isinstance(ckpt_path, Path):
            ckpt_path = ckpt_path.name
        with open(ckpt_path, "rb") as f:
            # grab from gcs or file system
            buffer = io.BytesIO(f.read()) # type: ignore
            checkpoint = torch.load(buffer, map_location="cpu")
        ckpts[int(ckpt_path.split('.', maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]
    
    params_path = checkpoint_dir + "/params.json"
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    jax_weights = {
        'transformer': {
            'wte': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].numpy() for ckpt in ckpts], axis=1)},
            'ln_f': {'kernel': ckpts[0]['norm.weight'].numpy()},
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': np.concatenate([ckpt['layers.%d.attention.wq.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()},
                        'wk': {'kernel': np.concatenate([ckpt['layers.%d.attention.wk.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()},
                        'wv': {'kernel': np.concatenate([ckpt['layers.%d.attention.wv.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()},
                        'wo': {'kernel': np.concatenate([ckpt['layers.%d.attention.wo.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()},
                    },
                    'feed_forward': {
                        'w1': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w1.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()},
                        'w2': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w2.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()},
                        'w3': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w3.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()},
                    },
                    'attention_norm': {'kernel': ckpts[0]['layers.%d.attention_norm.weight' % (layer)].numpy()},
                    'ffn_norm': {'kernel': ckpts[0]['layers.%d.ffn_norm.weight' % (layer)].numpy()},
                }
            for layer in range(params['n_layers'])},
        },
        'lm_head': {'kernel': np.concatenate([ckpt['output.weight'].numpy() for ckpt in ckpts], axis=0).transpose()},
    }
    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights, FLAGS.output_file
        )
    else:
        with mlxu.open_file(FLAGS.output_file, 'wb') as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))


if __name__ == '__main__':
    """
    python -m EasyLM.models.llama.convert_torch_to_easylm \
        --checkpoint_dir="gs://n2formal-public-data-europe/albert/llama/7B" \
        --output_file="gs://n2formal-public-data-europe/albert/llama/jaxllama/7B" \
        --streaming=True
    """
    mlxu.run(main)