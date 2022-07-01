from unet_utils import plot_imgs
import jax.numpy as jnp
import numpy as np
import jax


def plot_predictions(dataset, unet, unet_train_state, epoch):
    pred_logits = []
    pred_masks = []
    for test_img in dataset['test']['images']:
        logits_pred = unet.apply(
            {"params": unet_train_state.unet_train_state.params}, test_img.reshape((1,)+test_img.shape))[0]
        mask_pred = jnp.round(jax.nn.sigmoid(logits_pred))
        pred_logits.append(logits_pred)
        pred_masks.append(mask_pred)
    pred_masks = np.array(pred_masks)
    pred_logits = np.array(pred_logits)
    original_imgs = np.array(dataset['test']['images'])
    original_masks = np.array(dataset['test']['masks'])

    plot_imgs(
        original_imgs,
        original_masks,
        pred_logits=pred_logits,
        pred_masks=pred_masks,
        nm_img_to_plot=1,
        figsize=10,
        save_path=f"experiment-{get_date_string()}_epoch-{epoch}_prediction.png")
