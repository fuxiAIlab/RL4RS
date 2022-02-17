import sys
import pandas as pd
import pandasql as ps

dataset_file = sys.argv[1]
dataset_dir = sys.argv[2]

pysqldf = lambda q: ps.sqldf(q, globals())

# recsys15-click
if 'recsys15' in dataset_file:
    df = pd.read_csv(dataset_dir + '/yoochoose-clicks.dat', names=["sessionid", "timestamp", "item", "Category"])

    sql0 = """
    select sessionid, min(timestamp) as timestamp, item, Category
    from
    df a
    group by sessionid, item, substr(timestamp,1,12)
    """

    sql1 = """
    select a.*
    from
    df a
    join
    (SELECT item FROM df group by item having count(*)>=1000)b
    on a.item=b.item
    """

    sql2 = """
    select a.*
    from
    df2 a
    join
    (SELECT sessionid FROM df2 group by sessionid having count(*)>=13)b
    on a.sessionid=b.sessionid
    """

    sql3 = """
    select sessionid, group_concat(item) as items
    from
    df3
    group by sessionid
    order by timestamp asc
    """

    df = pysqldf(sql0)

    df2 = pysqldf(sql1)

    df3 = pysqldf(sql2)

    df4 = pysqldf(sql3)

    print('items num.', df3['item'].value_counts().count())
    print('max item id', df3['item'].max())
    print('sessionid num.', df3['sessionid'].value_counts().count())

    df4.to_csv(dataset_dir + '/' + dataset_file + '.csv', sep=' ', header=True, index=False, encoding='utf-8')

if 'movielens' in dataset_file:
    # movielens-25m
    df = pd.read_csv(dataset_dir + '/ml-25m/ratings.csv')
    # userId,movieId,rating,timestamp
    sql0 = """
    select *
    from
    df a
    where rating>=3
    """

    sql1 = """
    select a.*
    from
    df a
    join
    (SELECT movieId FROM df group by movieId having count(*)>=1000)b
    on a.movieId=b.movieId
    """

    sql2 = """
    select a.*
    from
    df2 a
    join
    (SELECT userId FROM df2 group by userId having count(*)>=30 and count(*)<=100)b
    on a.userId=b.userId
    """

    sql3 = """
    select userId as sessionid, group_concat(movieId) as items
    from
    df3
    group by userId
    order by timestamp asc
    """
    df = pysqldf(sql0)

    df2 = pysqldf(sql1)

    df3 = pysqldf(sql2)

    df4 = pysqldf(sql3)

    print('items num.', df3['movieId'].value_counts().count())
    print('max item id', df3['movieId'].max())
    print('sessionid num.', df3['userId'].value_counts().count())

    df4.to_csv(dataset_dir + '/movielens.csv', sep=' ', header=True, index=False, encoding='utf-8')

if 'rl4rs' in dataset_file:
    # RL4RS
    data = open(dataset_dir + '/rl4rs_dataset_a.csv', 'r').read().split('\n')[:-1]
    tmp = ['sessionid items']
    for x in data:
        session_id = x.split('@')[1]
        sequence_id = list(map(int, x.split('@')[5].split(',')))
        items = list(map(int, x.split('@')[3].split(',')))
        if len(sequence_id) >= 16:
            tmp.append(session_id + ' ' + ','.join(list(map(str, sequence_id[-16:] + items[:5]))))

    print('items num.', 283)
    print('max item id', 283)
    print('sessionid num.', len(tmp))

    with open(dataset_dir + '/rl4rs.csv', 'w') as f:
        f.write('\n'.join(tmp))
