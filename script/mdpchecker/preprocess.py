import sys
import pandas as pd
import pandasql as ps

dataset_file = sys.argv[1]
dataset_dir = sys.argv[2]

pysqldf = lambda q: ps.sqldf(q, globals())

# recsys15-click
if 'lastfm' in dataset_file:
    df = pd.read_csv(dataset_dir + '/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
                     ,names=["userid", "timestamp", "artid", "artname", "traid", "traname"]
                     ,sep='\t')

    sql0 = """
        select userid as sessionid, min(timestamp) as timestamp, artid
        from
        df a
        group by userid, artid, substr(timestamp,1,12)
        """

    sql1 = """
        select a.sessionid, a.timestamp, b.item
        from
        df a
        join
        (select artid, ROW_NUMBER() OVER(ORDER BY artid) AS item
            from (
                select artid
                from
                df a
                group by artid 
                having count(*)>=30
            )aa
        )b
        on a.artid=b.artid
        """

    sql2 = """
    select sessionid, group_concat(item) as items
    from(
        select *
        from
        df2
        order by timestamp asc
    )a
    group by sessionid
    
    """

    df = pysqldf(sql0)

    df2 = pysqldf(sql1)

    df3 = pysqldf(sql2)

    print('items num.', df2['item'].value_counts().count())
    print('max item id', df2['item'].max())
    print('sessionid num.', df2['sessionid'].value_counts().count())

    df3.to_csv(dataset_dir + '/' + dataset_file + '.csv', sep=' ', header=True, index=False, encoding='utf-8')

if 'cikm2016' in dataset_file:
    # queryId;sessionId;userId;timeframe;duration;eventdate;searchstring.tokens;categoryId;items;is.test
    queries_df = pd.read_csv(dataset_dir + '/CIKMCUP2016_Track2/train-queries.csv',sep=';')
    # queryId;timeframe;itemId
    click_df = pd.read_csv(dataset_dir + '/CIKMCUP2016_Track2/train-clicks.csv',sep=';')
    # sessionId;userId;itemId;timeframe;eventdate
    pv_df = pd.read_csv(dataset_dir + '/CIKMCUP2016_Track2/train-item-views.csv',sep=';')

    # sql0 = """
    #     select a.sessionId as sessionid, min(b.timeframe) as timestamp, b.itemId, a.items as pv_items
    #     from
    #     queries_df a
    #     join click_df b
    #     on a.queryId = b.queryId
    #     group by b.queryId, b.itemId, cast(b.timeframe/1000 as int)
    #     """

    df_click_sql = """
        select a.sessionId as sessionid, min(cast(b.timeframe as int)) as timestamp, b.itemId as item
        from
        queries_df a
        join click_df b
        on a.queryId = b.queryId
        join (select sessionId from pv_df group by sessionId)c 
        on a.sessionId = c.sessionId
        group by a.sessionId, b.itemId, cast(b.timeframe/1000 as int)
        """

    df_pv_sql = """
        select a.sessionId as sessionid, min(cast(c.timeframe as int)) as timestamp, c.itemId as item
        from
        queries_df a
        join (select queryId from click_df group by queryId) b
        on a.queryId = b.queryId
        join pv_df c 
        on a.sessionId = c.sessionId
        group by a.sessionId, c.itemId, cast(c.timeframe/1000 as int)
        """

    df_sql = """
    select aa.sessionid, group_concat(c.item|| ':' ||c.timestamp) as pv_items, aa.click_items
      from
        ( 
        select a.sessionid,a.timestamp,a.item,group_concat(b.item|| ':' ||b.timestamp) as click_items  from
        df_click a 
        join df_click b 
        on a.sessionid=b.sessionid and a.timestamp<=b.timestamp
        group by a.sessionid,a.item
        )aa
        
        join df_pv c 
        on aa.sessionid=c.sessionid and aa.timestamp>c.timestamp
        group by aa.sessionid,aa.click_items
    """

    df_click = pysqldf(df_click_sql)
    df_pv = pysqldf(df_pv_sql)
    df = pysqldf(df_sql)

    tmp = []
    items = set()
    for x in df.values:
        if len(x[1].split(','))>=5 and len(x[2].split(','))>=5:
            [items.add(x.split(':')[0]) for x in x[1].split(',')]
            [items.add(x.split(':')[0]) for x in x[2].split(',')]

    # item2id=dict([(x,str(i)) for i,x in enumerate(items)])
    # item2id_fn = lambda x:item2id[x]

    for x in df.values:
        if len(x[1].split(','))>=5 and len(x[2].split(','))>=5:
            pv_items = x[1].split(',')
            sorted_pv_items = sorted(pv_items, key=lambda x:int(x.split(':')[1]))[-5:]
            sorted_pv_items = [x.split(':')[0] for x in sorted_pv_items]
            click_items = x[2].split(',')
            sorted_click_items = sorted(click_items, key=lambda x:int(x.split(':')[1]))[:5]
            sorted_click_items = [x.split(':')[0] for x in sorted_click_items]
            tmp.append([x[0], ','.join(sorted_pv_items), ','.join(sorted_click_items)])

    print('items num.', len(items))
    print('max item id', len(items)-1)
    print('sessionid num.', len(tmp))

    with open(dataset_dir + '/' + dataset_file + '.csv', 'w') as f:
        f.write('sessionid items'+'\n')
        f.write('\n'.join([str(x[0])+' '+x[1]+','+x[2] for x in tmp]))


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
