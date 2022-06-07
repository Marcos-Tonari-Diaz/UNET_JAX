
import jax
import jax.numpy as jnp
import flax.linen as nn

# input dims [N: number of batches, H, W, C]


def center_crop_array(array, new_size):
    crop_offset = (array.shape[1] - new_size)//2
    crop_offset_low = crop_offset
    crop_offset_high = crop_offset if array.shape[1] % 2 == 0 else crop_offset+1
    return array[:, crop_offset_low:-crop_offset_high, crop_offset_low:-crop_offset_high, :]


def max_pool_block(input):
    return nn.max_pool(input, window_shape=(2, 2), strides=(2, 2))


class FinalBlock(nn.Module):
    use_padding: bool
    use_activation: bool
    default_weight_init = nn.initializers.he_normal

    def setup(self):
        # self.conv = nn.Conv(features=1, kernel_size=(
        #     1, 1), kernel_init=self.default_weight_init)
        self.conv = nn.Conv(features=1, kernel_size=(1, 1))
        self.activation = nn.sigmoid if self.use_activation else lambda x: x
        self.upscale = jax.image.resize if not self.use_padding else lambda x, shape, method: x

    def __call__(self, input):
        input = self.conv(input)
        input = self.activation(input)
        return self.upscale(input, shape=(1, 512, 512, 1), method='bilinear')


class ContractingBlock(nn.Module):
    num_features: int
    use_padding: bool
    default_weight_init = nn.initializers.he_normal

    def setup(self):
        if self.use_padding:
            self.conv_1 = nn.Conv(
                # features=self.num_features, kernel_size=(3, 3), kernel_init=self.default_weight_init)
                features=self.num_features, kernel_size=(3, 3))
            self.conv_2 = nn.Conv(
                # features=self.num_features, kernel_size=(3, 3), kernel_init=self.default_weight_init)
                features=self.num_features, kernel_size=(3, 3))
        else:
            self.conv_1 = nn.Conv(features=self.num_features,
                                  #   kernel_size=(3, 3), padding='VALID', kernel_init=self.default_weight_init)
                                  kernel_size=(3, 3), padding='VALID')
            self.conv_2 = nn.Conv(features=self.num_features,
                                  #   kernel_size=(3, 3), padding='VALID', kernel_init=self.default_weight_init)
                                  kernel_size=(3, 3), padding='VALID')

    @nn.compact
    def __call__(self, input):
        input = self.conv_1(input)
        input = nn.relu(input)
        input = self.conv_2(input)
        return nn.relu(input)


class ExpandingBlock(nn.Module):
    num_features: int
    use_padding: bool
    default_weight_init = nn.initializers.he_normal

    def setup(self):
        self.conv_tranpose = nn.ConvTranspose(
            # features=self.num_features, kernel_size=(2, 2), strides=(2, 2), kernel_init=self.default_weight_init)
            features=self.num_features, kernel_size=(2, 2), strides=(2, 2))
        if self.use_padding:
            self.conv_1 = nn.Conv(
                # features=self.num_features, kernel_size=(3, 3), kernel_init=self.default_weight_init)
                features=self.num_features, kernel_size=(3, 3))
            self.conv_2 = nn.Conv(
                # features=self.num_features, kernel_size=(3, 3), kernel_init=self.default_weight_init)
                features=self.num_features, kernel_size=(3, 3))
            self.crop = lambda x, y: x
        else:
            self.conv_1 = nn.Conv(features=self.num_features,
                                  #   kernel_size=(3, 3), padding='VALID', kernel_init=self.default_weight_init)
                                  kernel_size=(3, 3), padding='VALID')
            self.conv_2 = nn.Conv(features=self.num_features,
                                  #   kernel_size=(3, 3), padding='VALID', kernel_init=self.default_weight_init)
                                  kernel_size=(3, 3), padding='VALID')
            self.crop = center_crop_array

    @nn.compact
    def __call__(self, input, residual_feature_map):
        input = self.conv_tranpose(input)
        cropped_feature_map = self.crop(residual_feature_map, input.shape[1])
        input = jnp.concatenate((input, cropped_feature_map), axis=3)
        input = self.conv_1(input)
        input = nn.relu(input)
        input = self.conv_2(input)
        return nn.relu(input)


class UnetJAX(nn.Module):
    input_image_size: int
    use_padding: bool
    use_activation: bool

    def setup(self):

        self.contracting_block_1 = ContractingBlock(64, self.use_padding)
        self.contracting_block_2 = ContractingBlock(128, self.use_padding)
        self.contracting_block_3 = ContractingBlock(256, self.use_padding)
        self.contracting_block_4 = ContractingBlock(512, self.use_padding)
        self.contracting_block_5 = ContractingBlock(1024, self.use_padding)

        self.expanding_block_1 = ExpandingBlock(512, self.use_padding)
        self.expanding_block_2 = ExpandingBlock(256, self.use_padding)
        self.expanding_block_3 = ExpandingBlock(128, self.use_padding)
        self.expanding_block_4 = ExpandingBlock(64, self.use_padding)

        self.final_block = FinalBlock(self.use_padding, self.use_activation)

    def __call__(self, input):
        contracting_out1 = self.contracting_block_1(input)
        max_pool_out = max_pool_block(contracting_out1)
        contracting_out2 = self.contracting_block_2(max_pool_out)
        max_pool_out = max_pool_block(contracting_out2)
        contracting_out3 = self.contracting_block_3(max_pool_out)
        max_pool_out = max_pool_block(contracting_out3)
        contracting_out4 = self.contracting_block_4(max_pool_out)
        max_pool_out = max_pool_block(contracting_out4)
        contracting_out5 = self.contracting_block_5(max_pool_out)
        output = self.expanding_block_1(contracting_out5, contracting_out4)
        output = self.expanding_block_2(output, contracting_out3)
        output = self.expanding_block_3(output, contracting_out2)
        output = self.expanding_block_4(output, contracting_out1)
        output = self.final_block(output)
        return output

    def init_params(self, rng):
        input_size_dummy = jnp.ones(
            (1, self.input_image_size, self.input_image_size, 1))
        params = self.init(rng, input_size_dummy)
        return params


if __name__ == "__main__":
    from jax import random
    unet_padding = UnetJAX(input_image_size=512,
                           use_padding=True, use_activation=False)
    key = random.PRNGKey(0)
    unet_params = unet_padding.init_params(key)
    dummy_in = jnp.ones([1, 512, 512, 1])
    dummy_out = unet_padding.apply(unet_params, dummy_in)
    print("padding " + str(dummy_out.shape))
    unet_resize = UnetJAX(input_image_size=512,
                          use_padding=False, use_activation=False)
    unet_params = unet_resize.init_params(key)
    dummy_in = jnp.ones([1, 512, 512, 1])
    dummy_out = unet_resize.apply(unet_params, dummy_in)
    print("resize " + str(dummy_out.shape))
