import time
import os
import psutil
import multiprocessing as mp
import pickle


def process_line(line, mid2name):
    s, p, name, _ = line.strip().split('\t')
    
    mid = s[s.rfind('/') + 1: -1]

    if p == '<http://rdf.freebase.com/ns/type.object.name>' and mid[:2] in ('m.', 'g.'):
        if mid in mid2name:
            if name[-2:] == 'en':
                return mid, name
        else:
            return mid, name

    return None


def chunkify_file(filepath, num_chunks, skiplines=-1):
    chunks = []
    file_end = os.path.getsize(filepath)
    print(f'file size : {file_end}')
    size = file_end // num_chunks

    with open(filepath, "rb") as f:
        if(skiplines > 0):
            for i in range(skiplines):
                f.readline()

        chunk_end = f.tell()
        count = 0
        while True:
            chunk_start = chunk_end
            f.seek(f.tell() + size, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunk_end = f.tell()
            chunks.append((chunk_start, chunk_end - chunk_start, filepath))
            count += 1

            if chunk_end >= file_end:
                break

    assert len(chunks) == num_chunks

    return chunks

def parallel_apply_line_by_line_chunk(chunk_data):
    chunk_start, chunk_size, file_path, func_apply = chunk_data[:4]
    func_args = chunk_data[4:]

    print(f'start {chunk_start}')

    i = 0
    st = time.time()

    mid2name = dict()
    func_args.append(mid2name)

    with open(file_path, "rb") as f:
        f.seek(chunk_start)

        while True:
            i += 1

            line = f.readline().decode(encoding='utf-8')

            if line == '':
                # the last chunk of file ends with ''
                break

            ret = func_apply(line, *func_args)

            if(ret != None):
                mid, name = ret
                mid2name[mid] = name

            if i % 1_000_000 == 0:
                ed = time.time()
                print(ed - st, f.tell() - chunk_start, '/', chunk_size, (f.tell() - chunk_start) / chunk_size)
                st = ed

            if f.tell() - chunk_start >= chunk_size:
                break


    return mid2name

def parallel_apply_line_by_line(input_file_path, num_procs, func_apply, func_args, skiplines=0, fout=None, merge_func=None):
    num_parallel = num_procs
    print(f'num parallel: {num_procs}')

    jobs = chunkify_file(input_file_path, num_procs, skiplines)

    jobs = [list(x) + [func_apply] + func_args for x in jobs]

    print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    pool = mp.Pool(num_parallel, maxtasksperchild=1000)  # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering

    outputs = []

    t1 = time.time()
    chunk_outputs = pool.map(parallel_apply_line_by_line_chunk, jobs)

    for i, output in enumerate(chunk_outputs):
        outputs.append(output)

    pool.close()
    pool.terminate()

    if merge_func is not None:
        print('merging outputs...')
        output = merge_func(outputs)
    else:
        output = outputs

    print("All Done in time ", time.time() - t1)

    return output

def merge_mid2name(mid2names):
    final_mid2name = {}

    for mid2name in mid2names:
        for mid, name in mid2name.items():
            if mid in final_mid2name:
                if name[-2:] == 'en':
                    final_mid2name[mid] = name
                else:
                    continue
            else:
                final_mid2name[mid] = name

    return final_mid2name


if __name__ == '__main__':
    n_processes = 40

    mid2name = parallel_apply_line_by_line('/data/knowledge_graph/freebase/freebase-rdf-latest', n_processes, process_line, [], fout=None, merge_func=merge_mid2name)

    mid2name_path = '/data/knowledge_graph/freebase/mid2name_.pkl'
    with open(mid2name_path, 'wb') as f:
        pickle.dump(mid2name, f)
