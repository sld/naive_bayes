require 'naive_bayes'

describe "NaiveBayes" do 
  context "Multinomial" do
    before(:each) do
      @training_data = ["Chinese Beijing Chinese",
                                          "Chinese Chinese Shanghai",
                                          "Chinese Macao",
                                          "Tokyo Japan Chinese"]

      @test_data = ["Chinese Chinese Chinese Tokyo Japan"]


      # Classifier sets each word as feature by default
      @nb = NaiveBayes::NaiveBayes.new
      @training_data[0...3].each{ |data| @nb.train( data, :c ) }
      @nb.train( @training_data[3], :j )      
    end


    it "should correctly calculate conditional probabilites" do
      @nb.cond_prob( "Chinese", :c ).should == 3.0/7
      @nb.cond_prob( "Tokyo", :c ).should == 1.0/14
      @nb.cond_prob( "Tokyo", :j ).should == 2.0/9
      @nb.cond_prob( "Chinese", :j ).should == 2.0/9
    end


    it "should classify to return maximum class probablity :c" do
        @nb.classify( @test_data[0] )[:class].should == :c
    end


    it "should correctly calculate probablity of class" do
      @nb.class_prob(:c).should == 3.0/4
      @nb.class_prob(:j).should == 1.0/4      
    end


    it "should return ~=correctly probabilites for test_data" do
      features_vector = @nb.form_features_vector(@test_data[0])
      @nb.document_class_prob( features_vector, :c ).should > 0.0003
      @nb.document_class_prob( features_vector, :c ).should < 0.0004
      @nb.document_class_prob( features_vector, :j ).should > 0.0001
      @nb.document_class_prob( features_vector, :j ).should < 0.0002
    end
  end

  context "ROSE" do 
    before(:each) do
      @training_data = ["Chinese Beijing Chinese",
                                          "Chinese Chinese Shanghai",
                                          "Chinese Macao",
                                          "Tokyo Japan Chinese"]

      @test_data = ["Chinese Chinese Chinese Tokyo Japan"]

      
      @nb = NaiveBayes::NaiveBayes.new 1.0, :rose, {:rose => {:duplicate_count => 2, :duplicate_klass => :j} }

      @training_data[0...3].each{ |data| @nb.train( data, :c ) }
      @nb.train( @training_data[3], :j )      
    end

    it "should classify to return maximum class probablity :c.  But maybe it should be like that." do
      # It is strange that result is different from Multionimal approach. But maybe it should be like that.
      @nb.classify( @test_data[0] )[:class].should == :c
    end

  end


end