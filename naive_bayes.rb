require 'set'

class NaiveBayes


	# By default each part of string divided by space(" ") is feature
	def initialize
		@klasses_count = {}
		@word_classes_count = {}		
		@vocabolary = Set.new
	end


	def train( string, klass )
		tokens = get_features( string )
		@word_classes_count[klass] ||= {}			
		tokens.each do |token|		
			@vocabolary << token
		  @word_classes_count[klass][token] = @word_classes_count[klass][token].to_i + 1 
		end
		@klasses_count[klass] = @klasses_count[klass].to_i + 1		
	end


	def document_class_prob( string, klass )
		tokens = get_features( string )
		product_of_cond_probs = tokens.inject(1){ |product, e| product *= cond_prob( e, klass ) }
		class_prob( klass ) * product_of_cond_probs
	end


	def cond_prob( token, klass )		
		all_words_in_klass = @word_classes_count[klass].values.inject{ |e,s| s += e }
		( @word_classes_count[klass][token].to_i + 1.0 ) / ( all_words_in_klass + @vocabolary.count )
	end


	def class_prob( klass )
		@klasses_count[klass].to_f / @klasses_count.values.inject{ |e,s| s += e }
	end


	protected	


	def get_features( string )
		string.split(" ")
	end

end