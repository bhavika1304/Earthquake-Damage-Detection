from skimage.exposure import match_histograms


def apply_histogram_matching(source, reference):

    matched = match_histograms(
        source,
        reference,
        channel_axis=-1
    )

    return matched