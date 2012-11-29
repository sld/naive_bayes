require 'set'


class NaiveBayes


	# By default each part of string divided by space(" ") is feature
	def initialize
		# Class' documents count 
		# Example { :japanese => 3 } means that 3 documents with class :japanese
		@klass_docs_count = {}     
		# Class' words count	
		# Example: { :japanese => {"Tokyo" => times} } means that in class :japanese, word "Tokyo" was 3 times
		@klass_words_count = {}
		@vocabolary = Set.new
	end


	def train( string, klass )
		tokens = get_features( string )
		@klass_words_count[klass] ||= {}			
		tokens.each do |token|		
			@vocabolary << token
		  @klass_words_count[klass][token] = @klass_words_count[klass][token].to_i + 1 
		end
		@klass_docs_count[klass] = @klass_docs_count[klass].to_i + 1		
	end


	def classify( string )
		klasses = @klass_docs_count.keys
		klass_probs = {}
		klasses.each{ |klass| klass_probs[klass] = document_class_prob( string, klass ) }
		max_klass_prob = klass_probs.max_by{ |key, value| value }
		{:class => max_klass_prob[0], :value => max_klass_prob[1]}
	end


	def document_class_prob( string, klass )
		tokens = get_features( string )
		product_of_cond_probs = tokens.inject(1){ |product, e| product *= cond_prob( e, klass ) }
		class_prob( klass ) * product_of_cond_probs
	end


	def cond_prob( token, klass )		
		all_words_in_klass = @klass_words_count[klass].values.inject{ |e,s| s += e }
		( @klass_words_count[klass][token].to_i + 1.0 ) / ( all_words_in_klass + @vocabolary.count )
	end


	def class_prob( klass )
		@klass_docs_count[klass].to_f / @klass_docs_count.values.inject{ |e,s| s += e }
	end


	protected	


	def get_features( string )
		string.split(" ")
	end

end