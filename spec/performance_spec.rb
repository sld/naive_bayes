require 'naive_bayes'

def form_document_class_structure 
	dictionary = File.open("spec/data/la12.mat.clabel").each_line.to_a
	klasses = File.open("spec/data/la12.mat.rclass").each_line.to_a

	meta_data = nil
	feature_vectors = {}
	File.open("spec/data/la12.mat").each_line.each_with_index do |line, ind|
		if ind == 0
			meta_data = line.chomp 
			next 
		end

		feature_vector = []
		line.split(" ").each_slice(2) do |slice| 
			feature_vector << dictionary[slice[0].to_i-1].chomp*slice[1].to_i 
		end
		feature_vectors[klasses[ind-1].chomp] ||= []
		feature_vectors[klasses[ind-1].chomp] << feature_vector
	end

	return feature_vectors
end


def prepare_for_naive_bayes
	class_documents = form_document_class_structure
	minor_class = class_documents.min_by{|k,v| v.count}[0]
	all_other_classes = class_documents.keys - [minor_class]
	
	prepared_class_documents = { :minor => class_documents[minor_class].shuffle, 
															 :all_other => all_other_classes.inject([]){ |arr,kl| arr += class_documents[kl] }.shuffle }
  
  ao_half_count = prepared_class_documents[:all_other].count/2
  m_half_count = prepared_class_documents[:minor].count/2
  train_data = { :minor => prepared_class_documents[:minor][0...m_half_count],
  						   :all_other => prepared_class_documents[:all_other][0...ao_half_count] }
  test_data = { :minor => prepared_class_documents[:minor][m_half_count..-1],
  						   :all_other => prepared_class_documents[:all_other][ao_half_count..-1] }

  return [train_data, test_data]  
end


def naive_bayes_performance
	train_data, test_data = prepare_for_naive_bayes
	# @nb = NaiveBayes::NaiveBayes.new

	# p "Training"
	# train_data.each do |klass_name, klass_train_vectors|
	# 	klass_train_vectors.each do |train_vector|
	# 		@nb.train train_vector, klass_name
	# 	end
	# end

	# data = Marshal.dump @nb
	@nb = Marshal.load( File.open('nb_dumped', 'r') )

	p "Testing"
	confusion_matrix = { :minor => {:minor => 0, :all_other => 0}, :all_other => {:all_other => 0, :minor => 0} }

	test_data.each do |klass_name, klass_test_vectors|
		klass_test_vectors.each_with_index do |test_vector, i|
			puts "#{i}/#{klass_test_vectors.count}"			
			klass = @nb.classify(test_vector)[:class]
			confusion_matrix[klass_name][klass] += 1
		end
	end

	return confusion_matrix
end


@dataset_filename = 'dataset'
def save_dataset_to_file
	dataset = Marshal.dump( prepare_for_naive_bayes	)
	File.open(@dataset_filename, 'w').write(dataset)
end

def load_dataset_from_file
	Marshal.load File.open(@dataset_filename, 'r')
end


def naive_bayes_performance_rose
	train_data, test_data = load_dataset_from_file
	
	# @nb = NaiveBayes::NaiveBayes.new 1.0, :rose, :rose => { :duplicate_klass => :minor, 
	# 																			:duplicate_count => train_data[:all_other].count - train_data[:minor].count }

	# p "Training"
	# train_data.each do |klass_name, klass_train_vectors|
	# 	klass_train_vectors.each do |train_vector|			
	# 		@nb.train train_vector, klass_name
	# 	end
	# end

	p "Testing"
	# @nb.precache!
	# data = Marshal.dump @nb
	# File.open('nb_rose_dumped', 'w'){|f| f.write data}
	@nb = Marshal.load( File.open('nb_rose_dumped', 'r') )	
	confusion_matrix = { :minor => {:minor => 0, :all_other => 0}, :all_other => {:all_other => 0, :minor => 0} }

	# #require 'ruby-prof'

	# #RubyProf.start
	puts "Testing"
	test_data.each do |klass_name, klass_test_vectors|
		klass_test_vectors.each_with_index do |test_vector, i|
			puts "#{i}/#{klass_test_vectors.count}"			
			klass = @nb.classify(test_vector)[:class]
			confusion_matrix[klass_name][klass] += 1
		end
	end
	# # result = RubyProf.stop
	# # printer = RubyProf::GraphHtmlPrinter.new(result)
	# # File.open('ruby-prof-cached.html','w'){|file| printer.print(file)}

	# return confusion_matrix
end


# puts naive_bayes_performance
naive_bayes_performance_rose






