require "naive_bayes/version"
require 'set'

module NaiveBayes


	class NaiveBayes
	  # By default each part of string divided by space(" ") is feature
	  def initialize
	    # Class' documents count 
	    # Example { :japanese => 3 } means that 3 documents with class :japanese
	    @klass_docs_count = {}     
	    # Class' words count  
	    # Example: { :japanese => {"Tokyo" => 3} } means that in class :japanese, word "Tokyo" was 3 times
	    @klass_words_count = {}
	    @vocabolary = Set.new
	  end


	  def train( string, klass )	
	  	tokens = form_features_vector( string )	  	
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
	    features_vector = form_features_vector( string )	
	    klasses.each{ |klass| klass_probs[klass] = document_class_prob( features_vector, klass ) }
	    get_necessary_klass( klass_probs )	   
	  end


	  def document_class_prob( features_vector, klass )
	    product_of_cond_probs = features_vector.inject(1){ |product, e| product *= cond_prob( e, klass ) }
	    class_prob( klass ) * product_of_cond_probs
	  end


	  # Logarithmic version to avoid Arithmetic_underflow
	  def log_document_class_prob( features_vector, klass )
	    log_sum_of_cond_probs = features_vector.inject(0){ |sum, e| sum += Math::log( cond_prob( e, klass ) ) } 
	    log_val = Math::log( class_prob( klass ) ) + log_sum_of_cond_probs    
	  end


	  def cond_prob( token, klass )   
	    all_words_in_klass = @klass_words_count[klass].values.inject{ |e,s| s += e }.to_i
	    ( @klass_words_count[klass][token].to_i + 1.0 ) / ( all_words_in_klass + @vocabolary.count )
	  end


	  def class_prob( klass )
	    @klass_docs_count[klass].to_f / @klass_docs_count.values.inject{ |e,s| s += e }
	  end


	  def export
	  	{ :docs_count => @klass_docs_count,
	  		:words_count => @klass_words_count,
	  		:vocabolary => @vocabolary }
    end


    def import!( klass_docs_count, klass_words_count, vocabolary )
      @klass_docs_count = klass_docs_count
      @klass_words_count = klass_words_count
      @vocabolary = vocabolary
    end


    def form_features_vector string 
    	case string
	  		when Array
	  			return string
	  		when String 
					return get_features( string )
			end    	
    end


    protected


	  def get_features( string )
	    string.split(" ")
	  end


	  def get_necessary_klass( klass_probs )
      klass_probs = Hash[klass_probs.sort_by{|k,v| v}.reverse]
	  	max_klass_prob = klass_probs.first
	    {:class => max_klass_prob[0], :value => max_klass_prob[1], :all_values => klass_probs.values}
	  end

	end
end
