import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

public class Wordcount {
    public static class DecreasingComparator extends Text.Comparator {
        @SuppressWarnings("rawtypes")
        public int compare(WritableComparable a, WritableComparable b){
            return -super.compare(a, b);
        }
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static class TokenizerMapper extends Mapper <Object, Text, Text, IntWritable>{
        
        
        private final static IntWritable one  = new IntWritable(1); // map输出的value
        private Text word = new Text();// map输出的key
        
        private boolean caseSensitive;    //是否大小写敏感
        private Configuration conf;   
        private BufferedReader fis;     // 保存文件输入流
        private Set<String> patternsToSkip = new HashSet<String>();  // 用来保存需过滤的关键词，从配置文件中读出赋值 

        public void setup(Context context) throws IOException, InterruptedException {
            conf = context.getConfiguration( );
            caseSensitive = conf.getBoolean ( "wordcount.case.sensitive",true);// 配置文件中的wordcount.case.sensitive功能是否打开
            if (conf.getBoolean ( "wordcount.skip.patterns" , true)){  // 配置文件中的wordcount.skip.patterns功能是否打开
                URI[ ] patternsURIs = Job.getInstance(conf).getCacheFiles();
                for (URI patternsURI: patternsURIs){ // 每一个patternsURI都代表一个文件
                    Path patternsPath = new Path(patternsURI.getPath());
                    String patternsFileName = patternsPath.getName( ).toString();
                    parseSkipFile(patternsFileName);
                }
            }
        }
        /**
         * 整个setup就做了两件事： 1.读取配置文件中的wordcount.case.sensitive，赋值给caseSensitive变量
         * 2.读取配置文件中的wordcount.skip.patterns，如果为true，将CacheFiles的文件都加入过滤范围
         */
        
        private void parseSkipFile(String fileName){
            try{
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while((pattern = fis.readLine())!=null){ // SkipFile的每一行都是一个需要过滤的pattern，例如\!
                    patternsToSkip.add(pattern);
                }
            }catch(IOException ioe){
                System.err.println("Caught exception while parsing the cached file " + StringUtils.stringifyException(ioe));
            }
        }
        
        
        static enum CountersEnum { INPUT_WORDS }    
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
            String line = (caseSensitive)? 
                value.toString():value.toString().toLowerCase();   //如果设置了大小写敏感，就保留原样，否则全转换成小写
            line = line.replaceAll("\\d+", "");                      //消去数字
            line = line.replaceAll("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]","");   
            for(String pattern:patternsToSkip){             // 将数据中所有满足patternsToSkip的pattern都过滤掉
                line = line.replaceAll(pattern,"");  
            }  
            
            StringTokenizer itr = new StringTokenizer(line); 
            while(itr.hasMoreTokens()){
                word.set(itr.nextToken());
                if(word.getLength()>=3){              //单词长度大于等于3的才收集
                    context.write(word,one);       //用context.write收集<key,value>对
                     // getCounter(String groupName, String counterName)计数器
                    Counter counter = context.getCounter(CountersEnum.class.getName(),CountersEnum.INPUT_WORDS.toString());
                    counter.increment(1);  
                }
            }
        }
    }

    //将单个文件的输出进行key和value对调
    public static class SortMapper extends Mapper<Object, Text, IntWritable, Text>{
        protected void map(Object key, Text value, Mapper<Object, Text, IntWritable, Text>.Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] keyValueStrings = line.split("\t");
            if (keyValueStrings.length != 2) {
                System.err.println("string format error!!!!!");
                return;
            }
            int outkey = Integer.parseInt(keyValueStrings[1]);
            String outvalue = keyValueStrings[0];
            context.write(new IntWritable(outkey), new Text(outvalue));
        }
    }

    //对于单个文章的词频统计的处理
    public static class SortAllMapper extends Mapper<Object, Text, Text, IntWritable>{
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException{
            String line=value.toString();
            Integer num_tmp=Integer.valueOf(line.split("\t")[1]);
            IntWritable num=new IntWritable(num_tmp);
            Text word=new Text();
            word.set(line.split("\t")[0].split("<")[0]);
            context.write(word, num);
        }
    }


    //单个文件词频统计（未排序）
    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class SortFileReducer extends Reducer<IntWritable, Text, Text, NullWritable>{
        //将结果输出到多个文件或多个文件夹
        private MultipleOutputs<Text,NullWritable> mos;
        //创建对象
        protected void setup(Context context) throws IOException,InterruptedException {
            mos = new MultipleOutputs<Text, NullWritable>(context);
        }
        //关闭对象
        protected void cleanup(Context context) throws IOException,InterruptedException {
            mos.close();
        }

        private Text result = new Text();
        private HashMap<String, Integer> map = new HashMap<>();

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            for(Text val: values){
                String docId = val.toString().split("<")[1];
                docId = docId.substring(0, docId.length()-4);
                docId = docId.replaceAll("-", "");
                String oneWord = val.toString().split("<")[0];
                int sum = map.values().stream().mapToInt(i->i).sum();
                int rank = map.getOrDefault(docId, 0);
                if(rank == 100){
                    continue;
                }
                else {
                    rank += 1;
                    map.put(docId, rank); //0->1, n->n+1
                }
                result.set(oneWord);
                String str=rank+": "+result+", "+key;
                mos.write(docId, new Text(str), NullWritable.get() );
            }
        }
    }

    //统计所有词频并进行排序
    public static class SortReducer extends Reducer<IntWritable, Text, Text, NullWritable>{
        private Text result = new Text();
        int rank=1;

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values,Context context)throws IOException, InterruptedException {
            for (Text value : values) {
                if(rank > 100)
                {
                    break;
                }
                result.set(value.toString());
                String str=rank+": "+result+", "+key;
                rank++;
                context.write(new Text(str),NullWritable.get());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();
        if ((remainingArgs.length != 2) && (remainingArgs.length != 5)) {
            System.err.println("Usage: wordcount <in> <out> [-skip punctuations skipPatternFile]");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(Wordcount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        List<String> otherArgs = new ArrayList<String>(); // 除了 -skip 以外的其它参数
        for (int i = 0; i < remainingArgs.length; ++i) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri()); // 将 -skip 后面的参数，即skip模式文件的url，加入本地化缓存中
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true); // 这里设置的wordcount.skip.patterns属性，在mapper中使用
            } else {
                otherArgs.add(remainingArgs[i]); // 将除了 -skip 以外的其它参数加入otherArgs中
            }
        }
        FileInputFormat.addInputPath(job, new Path(otherArgs.get(0)));
        Path tmpdir=new Path("singlefiletmp");
        FileOutputFormat.setOutputPath(job, tmpdir);
        if(job.waitForCompletion(true))
        {
            //新建一个job处理排序和输出格式
            Job sortJob = Job.getInstance(conf, "sort file");
            sortJob.setJarByClass(Wordcount.class);

            FileInputFormat.addInputPath(sortJob, tmpdir);

            //map后交换key和value
            sortJob.setMapperClass(SortMapper.class);
            sortJob.setReducerClass(SortFileReducer.class);

            Path singleFileDir = new Path("single-file-output" );
            FileOutputFormat.setOutputPath(sortJob, singleFileDir);

            List<String> fileNameList = Arrays.asList("shakespearealls11", "shakespeareantony23", "shakespeareas12",
                    "shakespearecomedy7", "shakespearecoriolanus24", "shakespearecymbeline17", "shakespearefirst51",
                    "shakespearehamlet25", "shakespearejulius26", "shakespeareking45", "shakespearelife54",
                    "shakespearelife55", "shakespearelife56", "shakespearelovers62", "shakespeareloves8",
                    "shakespearemacbeth46", "shakespearemeasure13", "shakespearemerchant5", "shakespearemerry15",
                    "shakespearemidsummer16", "shakespearemuch3", "shakespeareothello47", "shakespearepericles21",
                    "shakespearerape61", "shakespeareromeo48", "shakespearesecond52", "shakespearesonnets59",
                    "shakespearesonnets", "shakespearetaming2", "shakespearetempest4", "shakespearethird53",
                    "shakespearetimon49", "shakespearetitus50", "shakespearetragedy57", "shakespearetragedy58",
                    "shakespearetroilus22", "shakespearetwelfth20", "shakespearetwo18", "shakespearevenus60",
                    "shakespearewinters19");

            for (String fileName : fileNameList) {
                MultipleOutputs.addNamedOutput(sortJob, fileName, TextOutputFormat.class,Text.class, NullWritable.class);
            }

            sortJob.setOutputKeyClass(IntWritable.class);
            sortJob.setOutputValueClass(Text.class);
            //排序改写成降序
            sortJob.setSortComparatorClass(DecreasingComparator.class);

            if(sortJob.waitForCompletion(true))
            {
                Job allJob = Job.getInstance(conf, "all count");
                allJob.setJarByClass(Wordcount.class);

                FileInputFormat.addInputPath(allJob, tmpdir);

                //map后交换key和value
                allJob.setMapperClass(SortAllMapper.class);
                allJob.setCombinerClass(IntSumReducer.class);
                allJob.setReducerClass(IntSumReducer.class);
                allJob.setOutputKeyClass(Text.class);
                allJob.setOutputValueClass(IntWritable.class);
                Path Alltmpdir = new Path("alltmp" );
                FileOutputFormat.setOutputPath(allJob, Alltmpdir);

                if(allJob.waitForCompletion(true))
                {
                    Job sortJob1 = Job.getInstance(conf, "sort all");
                    sortJob1.setJarByClass(Wordcount.class);

                    FileInputFormat.addInputPath(sortJob1, Alltmpdir);

                    sortJob1.setMapperClass(SortMapper.class);
                    sortJob1.setReducerClass(SortReducer.class);
                    sortJob1.setMapOutputKeyClass(IntWritable.class);
                    sortJob1.setMapOutputValueClass(Text.class);
                    FileOutputFormat.setOutputPath(sortJob1, new Path(otherArgs.get(1)));
                    sortJob1.setOutputKeyClass(IntWritable.class);
                    sortJob1.setOutputValueClass(Text.class);

                    //排序改写成降序
                    sortJob1.setSortComparatorClass(DecreasingComparator.class);

                    System.exit(sortJob1.waitForCompletion(true) ? 0 : 1);
                }
            }
        }
    }
}