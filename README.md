# NaiveBayes

Multinomial Naive Bayes.

## Installation

Add this line to your application's Gemfile:

    gem 'naive_bayes'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install naive_bayes

## Usage

    @training_data = ["Chinese Beijing Chinese",
                                        "Chinese Chinese Shanghai",
                                        "Chinese Macao",
                                        "Tokyo Japan Chinese"]
    @test_str = "Chinese Chinese Chinese Tokyo Japan"

    # Classifier sets each word as feature by default( string.split(" ") )
    @nb = NaiveBayes::NaiveBayes.new
    @training_data[0...3].each{ |data| @nb.train( data, :c ) }
    @nb.train( @training_data[3], :j ) 
    classified = @nb.classify( @test_str ) #=> {:class => xx, :value => xx, :all_values => xx}
    classified[:class] # Should return maximum likelihood class

You can redefine get_features method to get specific features from text.

Also you can pass features vector instead of string into train or classify:
    @training_data = [["Chinese", "Beijing", "Chinese"],                                        
                      ["Chinese", "Macao"],
                      ["Tokyo", "Japan", "Chinese"]]  
    @test_vector = ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]
    @nb = NaiveBayes::NaiveBayes.new
    @training_data[0...2].each{ |data| @nb.train( data, :c ) } 
    @nb.train( @training_data[2], :j ) 
    @nb.classify( @test_vector )[:class] 


## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
