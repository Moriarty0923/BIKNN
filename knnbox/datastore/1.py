if not os.path.exists(output_filename + ".trained"):
    index_dim = pca_dim if do_pca else dimension
    quantizer = faiss.IndexFlatL2(index_dim)
    index = faiss.IndexIVFPQ(quantizer, index_dim, n_centroids, code_size, 8)
    index.nprobe = n_probe

    # 如果使用 GPU，直接在 GPU 上创建和训练索引
    if use_gpu:
        start = time.time()
        res = faiss.StandardGpuResources()
        co = faiss.GpuIndexIVFPQConfig()
        co.useFloat16 = True
        gpu_index = faiss.GpuIndexIVFPQ(res, index_dim, n_centroids, code_size, 8, co)
        gpu_index.nprobe = n_probe
        if verbose:
            print("  > [{}/{}] created index on GPU took {} s". \
                format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1

        if verbose:
            print("  > [{}/{}] training index (about 3 minutes)...".format(progress_idx, total_progress))
        start = time.time()
        np.random.seed(seed)
        random_sample = np.random.choice(
            np.arange(capacity), size=[min(train_index_count, capacity)],
            replace=False
        )

        # 训练索引
        gpu_index.train(keys[random_sample].astype(np.float32))

        if verbose:
            print("  > [{}/{}] training took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
            print("  > [{}/{}] writing index after training...".format(progress_idx, total_progress))
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename + ".trained")
        if verbose:
            print("  > [{}/{}] writing index took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
    else:
        if verbose:
            print("  > [{}/{}] training index (about 3 minutes)...".format(progress_idx, total_progress))
        start = time.time()
        np.random.seed(seed)
        random_sample = np.random.choice(
            np.arange(capacity), size=[min(train_index_count, capacity)],
            replace=False
        )

        # 训练索引
        index.train(keys[random_sample].astype(np.float32))

        if verbose:
            print("  > [{}/{}] training took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
            print("  > [{}/{}] writing index after training...".format(progress_idx, total_progress))
        start = time.time()
        faiss.write_index(index, output_filename + ".trained")
        if verbose:
            print("  > [{}/{}] writing index took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
